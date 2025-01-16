import torch
from torch_geometric.datasets import Planetoid

def load_cora_data():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    return dataset

if __name__ == "__main__":
    print('Test')
    dataset = load_cora_data()
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
