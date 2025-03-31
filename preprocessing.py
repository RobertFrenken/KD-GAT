import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

def graph_creation(combined, path, datasize=1.0, window_size=50, stride=50):
    
    if combined:
        # simple BC where all datasets are combined
        path = r'datasets/Car-Hacking Dataset/Fuzzy_dataset.csv'
        df_Fuzzy = dataset_creation(path)
        
        path = r'datasets/Car-Hacking Dataset/DoS_dataset.csv'
        df_DoS = dataset_creation(path)
        
        path = r'datasets/Car-Hacking Dataset/gear_dataset.csv'
        df_gear = dataset_creation(path)
        
        path = r'datasets/Car-Hacking Dataset/RPM_dataset.csv'
        df_RPM = dataset_creation(path)
        
        list_graphs_fuzzy = create_graphs(df_Fuzzy, window_size=window_size, stride=stride)

        list_graphs_DoS = create_graphs(df_DoS, window_size=window_size, stride=stride)
        
        list_graphs_gear = create_graphs(df_gear, window_size=window_size, stride=stride)
        
        list_graphs_RPM = create_graphs(df_RPM, window_size=window_size, stride=stride)
        
        combined_list = list_graphs_fuzzy + list_graphs_DoS + list_graphs_gear + list_graphs_RPM
        # Create the dataset
        dataset = GraphDataset(combined_list)

    
    else:
        arr = dataset_creation(path)
        list_graphs = create_graphs(arr, window_size=50, stride=50)
        # Create the dataset
        dataset = GraphDataset(list_graphs)

    return dataset


def hex_to_decimal(x):
    if x is None or x == 'None':
        return None
    try:
        return int(x, 16)
    except (ValueError, TypeError):
        return x
    

def pad_row(row):
    if row['DLC'] != 8:
        # grab the label
        label = row['Data'+str(row['DLC']+1)]
        row['Data'+str(row['DLC']+1)] = '00'
        row['label'] = label

    row.fillna(value='00', inplace=True) # Fill missing values with '00'
    return row

def dataset_creation(path):
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'CAN ID','DLC','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8', 'label'] 

    df['Source'] = df['CAN ID']
    df['Target'] = df['CAN ID'].shift(-1)

    df = df.apply(pad_row, axis=1)

    df = df.apply(lambda x: x.apply(hex_to_decimal))

    # Drop the last row
    df = df.drop(df.index[-1])

    # reencode the labels
    df['label'] = df['label'].replace({'R': 0, 'T': 1})

    return df[['CAN ID', 'Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8', 'Source', 'Target', 'label']]

def create_graphs(data, window_size, stride):
    
    graphs = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i+window_size]
        graph = window_data_transform(window)
        graphs.append(graph)
    return graphs

def window_data_transform(df):
    # Calculate edge counts for each unique (Source, Target) pair
    edge_counts = df.groupby(['Source', 'Target']).size().reset_index(name='count')

    # Create node index mapping
    nodes = pd.unique(pd.concat([df['Source'], df['Target']]))
    node_to_idx = {n:i for i,n in enumerate(nodes)}

    # Convert to edge indices tensor
    edge_index = torch.tensor(
        edge_counts[['Source','Target']].apply(lambda x: x.map(node_to_idx)).values.T, 
        dtype=torch.long
    )

    # Create edge features tensor (counts)
    edge_attr = torch.tensor(edge_counts['count'].values, dtype=torch.float).view(-1,1)

    # Calculate node statistics
    node_features = df.groupby('CAN ID').agg(
    Data1=('Data1', 'mean'),
    Data2=('Data2', 'mean'),
    Data3=('Data3', 'mean'),
    Data4=('Data4', 'mean'),
    Data5=('Data5', 'mean'),
    Data6=('Data6', 'mean'),
    Data7=('Data7', 'mean'),
    Data8=('Data8', 'mean'),
    Count=('Source', 'count')).reindex(nodes).reset_index()  # Ensure index alignment

    # Create node features tensor
    x = torch.tensor(node_features.values, dtype=torch.float)
    y = torch.tensor([1 if 1 in df['label'] else 0], dtype=torch.long)

    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    return graph_data

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
