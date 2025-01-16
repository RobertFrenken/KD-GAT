import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import train_test_split_edges
from GNN_model import GNN

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.train_pos_edge_index)
    target = torch.ones(data.train_pos_edge_index.size(1), 1, device=out.device)
    loss = F.binary_cross_entropy(out, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.test_pos_edge_index)
        pred = (out > 0.5).float()
        target = torch.ones(data.test_pos_edge_index.size(1), 1, device=out.device)
        acc = (pred == target).sum().item() / target.size(0)
    return acc

def main():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    data = train_test_split_edges(data)

    model = GNN(input_dim=dataset.num_features, hidden_dim=16, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train(model, optimizer, data)
        if epoch % 10 == 0:
            acc = test(model, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

if __name__ == "__main__":
    main()
