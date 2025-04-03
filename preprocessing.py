import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

def graph_creation(combined, path, datasize=1.0, window_size=50, stride=50):
    
    if combined:
        # simple BC where all datasets are combined
        path = r'datasets/Car-Hacking Dataset/Fuzzy_dataset.csv'
        df_Fuzzy = dataset_creation_vectorized(path)
        
        path = r'datasets/Car-Hacking Dataset/DoS_dataset.csv'
        df_DoS = dataset_creation_vectorized(path)
        
        path = r'datasets/Car-Hacking Dataset/gear_dataset.csv'
        df_gear = dataset_creation_vectorized(path)
        
        path = r'datasets/Car-Hacking Dataset/RPM_dataset.csv'
        df_RPM = dataset_creation_vectorized(path)
        
        list_graphs_fuzzy = create_graphs_numpy(df_Fuzzy, window_size=window_size, stride=stride)

        list_graphs_DoS = create_graphs_numpy(df_DoS, window_size=window_size, stride=stride)
        
        list_graphs_gear = create_graphs_numpy(df_gear, window_size=window_size, stride=stride)
        
        list_graphs_RPM = create_graphs_numpy(df_RPM, window_size=window_size, stride=stride)
        
        combined_list = list_graphs_fuzzy + list_graphs_DoS + list_graphs_gear + list_graphs_RPM
        # Create the dataset
        dataset = GraphDataset(combined_list)

    
    else:
        arr = dataset_creation_vectorized(path)
        list_graphs = create_graphs_numpy(arr, window_size=50, stride=50)
        # Create the dataset
        dataset = GraphDataset(list_graphs)

    return dataset

def create_graphs_numpy(data, window_size, stride):
    """
    Transforms a pandas dataframe into a list of PyTorch Geometric Data object.
    
    Args:
        data (pd.dataframe): A NumPy array representing a window of data.
                              Assumes the following column structure:
                              [Source, Target, Data1, Data2, ..., DataN, label]
        window_size (int): The size of the sliding window.
        stride (int): The stride for the sliding window.
    
    Returns:
        graphs: A list of PyTorch Geometric Data objects.
    """
    # Calculate the number of windows
    data = data.to_numpy()  # Convert DataFrame to NumPy array if necessary
    num_windows = (len(data) - window_size) // stride + 1

    # Preallocate a list for graphs
    graphs = []

    # Use NumPy slicing to extract windows
    for i in range(num_windows):
        start_idx = i * stride
        window = data[start_idx:start_idx + window_size]

        # Transform the window into a graph
        graph = window_data_transform_numpy(window)
        graphs.append(graph)

    return graphs


def window_data_transform_numpy(data):
    """
    Transforms a NumPy array window into a PyTorch Geometric Data object.
    
    Args:
        data (numpy.ndarray): A NumPy array representing a window of data.
                              Assumes the following column structure:
                              [Source, Target, Data1, Data2, ..., DataN, label]
    
    Returns:
        Data: A PyTorch Geometric Data object.
    """
    # Extract Source, Target, and label columns
    source = data[:, 0]  # Assuming Source is the first column
    target = data[:, -2]  # Assuming Target is the second column
    labels = data[:, -1]  # Assuming label is the last column

    # Calculate edge counts for each unique (Source, Target) pair
    unique_edges, edge_counts = np.unique(np.stack((source, target), axis=1), axis=0, return_counts=True)

    # Create node index mapping
    nodes = np.unique(np.concatenate((source, target)))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Convert edges to indices
    edge_index = np.vectorize(node_to_idx.get)(unique_edges).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create edge features tensor (counts)
    edge_attr = torch.tensor(edge_counts, dtype=torch.float).view(-1, 1)

    # Calculate node features (mean of data columns for each node)
    node_features = np.zeros((len(nodes), 10))  # 9 features: ID + 8 data columns
    node_counts = np.zeros(len(nodes))  # To store the count of each node
    for i, node in enumerate(nodes):
        mask = (source == node) # Only consider rows where the node is the source
        node_data = data[mask, 0:9]  # Data columns are assumed to be from index 0 to 9 (exclusive of label)
        if len(node_data) > 0:  # Avoid empty slices
            node_features[i, :-1] = node_data.mean(axis=0)  # Calculate mean of data columns
        node_counts[i] = mask.sum()  # Count occurrences of the node

    # Append the node counts as the last feature
    node_features[:, -1] = node_counts
    # Create node features tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Create label tensor (binary classification: 1 if any label is 1, else 0)
    y = torch.tensor([1 if 1 in labels else 0], dtype=torch.long)

    # Create the graph data object
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    return graph_data


def hex_to_decimal_vectorized(series):
    return series.apply(lambda x: int(x, 16) if pd.notnull(x) and x != 'None' else None)

def pad_row_vectorized(df):
    mask = df['DLC'] != 8
    for i in range(8, 1, -1):  # Iterate from Data8 to Data1
        mask_label = mask & (df['DLC'] + 1 == i)
        df.loc[mask_label, 'label'] = df.loc[mask_label, f'Data{i}']
        df.loc[mask_label, f'Data{i}'] = '00'
    return df.fillna('00')

def dataset_creation_vectorized(path):
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'CAN ID', 'DLC', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8', 'label']

    df['Source'] = df['CAN ID']
    df['Target'] = df['CAN ID'].shift(-1)

    # Pad rows and fill missing values
    df = pad_row_vectorized(df)

    # Convert hex columns to decimal
    hex_columns = ['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target']
    df[hex_columns] = df[hex_columns].apply(hex_to_decimal_vectorized)

    # Drop the last row
    df = df.iloc[:-1]

    # Reencode the labels
    df['label'] = df['label'].replace({'R': 0, 'T': 1})

    return df[['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target', 'label']]

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
