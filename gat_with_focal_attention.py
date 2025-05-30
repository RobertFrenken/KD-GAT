import os, torch, numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import classification_report, precision_recall_curve
from models.models import GATWithJK
from preprocessing import graph_creation
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.nn import GATConv
from pathlib import Path


class GATPlain(GATWithJK):
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        from torch_geometric.nn import global_mean_pool
        return self.lin(global_mean_pool(x, batch)).view(-1)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, gamma=1.2):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()

# find the best threshold for all models
@torch.no_grad()
def find_best_threshold(model, loader, device):
    model.eval(); all_p, all_t = [], []
    for data in loader:
        data = data.to(device)
        prob = torch.sigmoid(model(data)).cpu()
        all_p.append(prob); all_t.append(data.y.cpu())
    probs = torch.cat(all_p).numpy(); targets = torch.cat(all_t).numpy()
    p, r, th = precision_recall_curve(targets, probs)
    f1 = 2*p*r/(p+r+1e-9)
    best_t = th[np.argmax(f1)] if len(th) > 0 else 0.5
    return float(best_t)

# evaluation
@torch.no_grad()
def evaluate(model, loader, device, tag, thresh=0.5):
    model.eval(); trues, preds = [], []
    for data in loader:
        data = data.to(device)
        pr = (torch.sigmoid(model(data)) > thresh).int().cpu()
        trues.append(data.y.cpu()); preds.append(pr)
    trues = torch.cat(trues).numpy(); preds = torch.cat(preds).numpy()
    print(f"\n Evaluation ({tag}, thr={thresh:.3f})")
    print(classification_report(trues, preds, target_names=["Normal","Attack"]))

# train

def train_one_set(root, device, epochs=4, batch_size=64, save_dir="trained_model"):
    ds = graph_creation(root)
    train_len = int(len(ds)*0.8)
    train_ds, val_ds = random_split(ds, [train_len, len(ds)-train_len],
                                    generator=torch.Generator().manual_seed(42))
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds,  batch_size=batch_size)

    model = GATPlain(in_channels=10, hidden_channels=64, out_channels=1,
                     num_layers=3, heads=4).to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=5e-4)
    crit = FocalLoss(alpha=0.8, gamma=1.5)

    print(f"\nTraining on {Path(root).name}  (Train:{len(train_ds)}, Val:{len(val_ds)})")
    for ep in range(1, epochs+1):
        model.train(); loss_ep = 0
        for data in train_ld:
            data = data.to(device)
            opt.zero_grad(); loss = crit(model(data), data.y.float())
            loss.backward(); opt.step(); loss_ep += loss.item()
        print(f"  Epoch {ep}/{epochs} loss={loss_ep/len(train_ld):.4f}")
    best_thresh = find_best_threshold(model, val_ld, device)
    print(f"  ↳ Best threshold on val = {best_thresh:.3f}")
    evaluate(model, val_ld, device, tag=f"{Path(root).name} ->self‑holdout", thresh=best_thresh)
    os.makedirs(save_dir, exist_ok=True)
    mpath = os.path.join(save_dir, f"gat_focal_{Path(root).name}.pth")
    torch.save({"model": model.state_dict(), "thresh": best_thresh}, mpath)
    print(f"   model+thresh saved → {mpath}")
    return model, best_thresh, val_ld


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_BASE = "/users/PAS1266/graffz/Dataset/can-train-and-test-v1.5"
    SETS = ["set_01", "set_02", "set_03", "set_04"]
    loaders_cache, model_info = {}, {}

    for s in SETS:
        root = f"{ROOT_BASE}/{s}"
        mdl, thr, hold_ld = train_one_set(root, DEVICE)
        model_info[s] = (mdl, thr); loaders_cache[s] = hold_ld
    for train_set, (mdl, thr) in model_info.items():
        print(f"\n========== Cross‑test | model trained on {train_set} (thr={thr:.3f}) ==========")
        for test_set in SETS:
            if test_set == train_set:  
                continue
            if test_set not in loaders_cache:
                loaders_cache[test_set] = DataLoader(
                    graph_creation(f"{ROOT_BASE}/{test_set}"), batch_size=64)
            evaluate(mdl, loaders_cache[test_set], DEVICE,
                     tag=f"{train_set} -> {test_set}", thresh=thr)