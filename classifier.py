# ============================================================import os, json, torch, numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import global_mean_pool
from preprocessing import graph_creation#try solely id later               
from models.models import GATWithJK          
try:
    from gat_with_focal_attention import GATPlain      
except ImportError:
    class GATPlain(GATWithJK):             
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.lin(x).view(-1)


ROOT_BASE= "/users/PAS1266/graffz/Dataset/can-train-and-test-v1.5"
MODEL_DIR= "/users/PAS1266/graffz/CAN-Graph/trained_model"   
SETS= ["set_01", "set_02", "set_03", "set_04"]

BATCH_SIZE= 64
EPOCHS= 6
LR= 5e-4
HIDDEN= 64
HEADS= 4
NUM_LAYERS= 3
DROPOUT= 0.5
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(MODEL_DIR, exist_ok=True)

#create dataset
domain_map = {s: idx for idx, s in enumerate(SETS)}

def build_meta_dataset():
    from torch_geometric.data import Dataset
    all_graphs = []
    for s in SETS:
        graphs = graph_creation(os.path.join(ROOT_BASE, s))
        dom_id  = domain_map[s]#set01->0
        for g in graphs:
            g.attack_y = g.y.clone()                      
            g.y= torch.tensor([dom_id])           
            all_graphs.append(g)

    class _GD(Dataset):
        def __init__(self, data_list): self.data_list=data_list
        def __len__(self): return len(self.data_list)
        def __getitem__(self, idx): return self.data_list[idx]
    return _GD(all_graphs)

meta_ds= build_meta_dataset()
train_len = int(0.8*len(meta_ds))
train_ds, val_ds = random_split(meta_ds, [train_len, len(meta_ds)-train_len],
                                generator=torch.Generator().manual_seed(42))
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_ld= DataLoader(val_ds,  batch_size=BATCH_SIZE)

# ---------- 2. Meta-Classifier ----------
class GATDomainClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 heads, num_domains, dropout):
        super().__init__()
        self.backbone = GATWithJK(in_channels, hidden_channels,
                                  out_channels=num_domains,
                                  num_layers=num_layers,
                                  heads=heads, dropout=dropout)
    def forward(self, data): return self.backbone(data)

in_dim = meta_ds[0].x.size(1)
meta_model = GATDomainClassifier(in_dim, HIDDEN, NUM_LAYERS, HEADS,
                                 num_domains=len(SETS), dropout=DROPOUT).to(DEVICE)
optim= torch.optim.Adam(meta_model.parameters(), lr=LR)
ce_loss = torch.nn.CrossEntropyLoss()

#test classifier
from sklearn.metrics import classification_report, confusion_matrix

def eval_domain(loader, verbose=False):
    meta_model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(DEVICE)
            out = meta_model(d)#vector of 4
            ys.extend(d.y.cpu().numpy())
            preds.extend(out.argmax(1).cpu().numpy())
    acc = (np.array(ys)==np.array(preds)).mean()
    if verbose:
        print("\n Meta-Classifier per-set report")
        print(classification_report(ys, preds, target_names=SETS, digits=4))
        print("Confusion-matrix (rows=truth, cols=pred):")
        print(confusion_matrix(ys, preds))
    return acc


print("\n Training Meta-Classifier …")
best_acc = 0.0
for ep in range(1, EPOCHS+1):
    meta_model.train(); tot=0
    for d in train_ld:
        d = d.to(DEVICE)
        optim.zero_grad()
        loss = ce_loss(meta_model(d), d.y)
        loss.backward(); optim.step(); tot += loss.item()
    val_acc = eval_domain(val_ld)
    best_acc = max(best_acc, val_acc)
    print(f"Epoch {ep}/{EPOCHS}  train_loss={tot/len(train_ld):.4f}  Val-Acc={val_acc:.4f}")

print(f"\n Best Domain-Acc (Meta) = {best_acc:.4f}")
eval_domain(val_ld, verbose=True)

#save
meta_path = os.path.join(MODEL_DIR, "meta_classifier.pth")
torch.save(meta_model.state_dict(), meta_path)
with open(os.path.join(MODEL_DIR, "domain_map.json"), "w") as f:
    json.dump(domain_map, f)
print(f"Meta-Classifier saved →  {meta_path}")

def load_detection_models():
    models_dict, thresh_dict = {}, {}
    for s in SETS:
        ckpt_path = os.path.join(MODEL_DIR, f"gat_focal_{s}.pth")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        det = GATPlain(in_channels=in_dim, hidden_channels=64,
                       out_channels=1, num_layers=3, heads=4).to(DEVICE)
        det.load_state_dict(ckpt["model"]); det.eval()
        models_dict[s]   = det
        thresh_dict[s]   = ckpt["thresh"]
    return models_dict, thresh_dict
 
models_dict, thresh_dict = load_detection_models()

# test on distributing
def route_graph(g):
    g = g.to(DEVICE)
    with torch.no_grad():
        dom_id = meta_model(g).argmax(1).item()#choose the most similar graph
        set_name = SETS[dom_id]
        det= models_dict[set_name]
        thr= thresh_dict[set_name]
        yhat = (torch.sigmoid(det(g)) > thr).int().cpu().item()
    return yhat

y_true, y_pred = [], []
meta_model.eval()
for g in val_ds:
    y_true.append(int(g.attack_y))
    y_pred.append(route_graph(g))

print("\n Pipeline on Val-Set (Meta-Routing + Detection)")
print(classification_report(y_true, y_pred, target_names=["Normal","Attack"]))