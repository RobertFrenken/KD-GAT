import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

def graph_creation(combined, path, datasize=1.0, window_size=50, stride=50):
    
    if combined:
        # simple BC where all datasets are combined
        path = r'datasets/Car-Hacking Dataset/Fuzzy_dataset.csv'
        arr_Fuzzy = dataset_creation(path)
        
        path = r'datasets/Car-Hacking Dataset/DoS_dataset.csv'
        arr_DoS = dataset_creation(path)
        
        path = r'datasets/Car-Hacking Dataset/gear_dataset.csv'
        arr_gear = dataset_creation(path)
        
        path = r'datasets/Car-Hacking Dataset/RPM_dataset.csv'
        arr_RPM = dataset_creation(path)
        
        list_graphs_fuzzy = create_graphs(arr_Fuzzy, window_size=window_size, stride=stride)

        list_graphs_DoS = create_graphs(arr_DoS, window_size=window_size, stride=stride)
        
        list_graphs_gear = create_graphs(arr_gear, window_size=window_size, stride=stride)
        
        list_graphs_RPM = create_graphs(arr_RPM, window_size=window_size, stride=stride)
        
        combined_list = list_graphs_fuzzy + list_graphs_DoS + list_graphs_gear + list_graphs_RPM
        # Create the dataset
        dataset = GraphDataset(combined_list)

    
    else:
        arr = dataset_creation(path)
        list_graphs = create_graphs(arr, window_size=50, stride=50)
        # Create the dataset
        dataset = GraphDataset(list_graphs)

    return dataset


def dataset_creation(path):
    '''
    This function takes in a pandas dataframe, creates a Node and Edge column, 
    converts the hex values to decimal values, changes the label to a binary value,
    replaces NaN values with zero, and returns a numpy array.
    '''
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'CAN ID','DLC','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8', 'label'] 

    df['Node'] = df['CAN ID']
    df['Edge'] = df['CAN ID'].shift(-1)

    df = df.apply(lambda x: x.apply(hex_to_decimal))

    # Drop the last row
    df = df.drop(df.index[-1])

    # Drop rows where 'Data3' contains 'R'
    # I need to make this more comprehensive
    df_dropped = df[df['Data3'] != 'R']
    df= df_dropped

    # reencode the labels
    df['label'] = df['label'].replace({'R': 0, 'T': 1})

    # Replace NaN values with zero
    df = df.fillna(0)

    # drop timestamp here, and keep everything else for the numpy array
    df = df.drop(columns=['Timestamp'])

    arr = df.to_numpy(dtype=float)
    # arr = df[['Node', 'Edge', 'label']].to_numpy(dtype=float)

    return arr

def hex_to_decimal(x):
        if x is None or x == 'None':
            return None
        try:
            return int(x, 16)
        except (ValueError, TypeError):
            return x

def create_graphs(data, window_size, stride):
    graphs = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i+window_size]
        graph = create_single_graph(window)
        graphs.append(graph)
    return graphs


def create_node_features(data):
    # What does the node feature matrix look like?
    # [ID, occurance count, ]
    # I should make this a function
    # Node features:
    # - Node ID
    # - Frequency in sliding window
    # - average data field value
    unique, counts = np.unique(data[:, 0:1], return_counts=True)
    result = dict(zip(unique, counts))
    
    x = torch.tensor(data[:, 0:1], dtype=torch.float)
    return x

def create_single_graph(window_data):
        
        x = create_node_features(window_data)
        
        edge_index = _get_edge_index(window_data) # call the edge index function here
        
        label = window_data[:, -1] # last column are the labels
        y = torch.tensor([1 if np.any(label == 1) else 0], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

def _get_edge_index(window_data: np.ndarray) -> torch.Tensor:
        # ------------------------
        # Edge features:
        # - number of connections/ instances between the two CAN IDs
        num_nodes = window_data.shape[0]
        edge_index = torch.tensor([np.arange(num_nodes - 1), np.arange(1, num_nodes)], dtype=torch.long)
        return edge_index

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
