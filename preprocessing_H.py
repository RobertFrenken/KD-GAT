import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

def graph_creation(combined, path, datasize=1.0, window_size=50, stride=50):
   
    if combined:

        fuzzy_path = r'D:/quadeer/task1/dataset/Fuzzy_dataset.csv'
        dos_path   = r'D:/quadeer/task1/dataset/DoS_dataset.csv'
        gear_path  = r'D:/quadeer/task1/dataset/gear_dataset.csv'
        rpm_path   = r'D:/quadeer/task1/dataset/RPM_dataset.csv'

        df_Fuzzy = dataset_creation_vectorized(fuzzy_path)
        df_DoS   = dataset_creation_vectorized(dos_path)
        df_gear  = dataset_creation_vectorized(gear_path)
        df_RPM   = dataset_creation_vectorized(rpm_path)

        list_graphs_fuzzy = create_graphs_numpy(df_Fuzzy, window_size=window_size, stride=stride)
        list_graphs_DoS   = create_graphs_numpy(df_DoS,   window_size=window_size, stride=stride)
        list_graphs_gear  = create_graphs_numpy(df_gear,  window_size=window_size, stride=stride)
        list_graphs_RPM   = create_graphs_numpy(df_RPM,   window_size=window_size, stride=stride)

        combined_list = list_graphs_fuzzy + list_graphs_DoS + list_graphs_gear + list_graphs_RPM
        dataset = GraphDataset(combined_list)
    else:
     
        df = dataset_creation_vectorized(path)
        list_graphs = create_graphs_numpy(df, window_size=window_size, stride=stride)
        dataset = GraphDataset(list_graphs)

    return dataset

def dataset_creation_vectorized(path):


    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'CAN ID', 'DLC', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8', 'label']

   
    df['Source'] = df['CAN ID']
    df['Target'] = df['CAN ID'].shift(-1)


    df = pad_row_vectorized(df)


    hex_columns = ['CAN ID','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8','Source','Target']
    df[hex_columns] = df[hex_columns].apply(hex_to_decimal_vectorized)


    df = df.iloc[:-1]

    df['label'] = df['label'].replace({'R': 0, 'T': 1})

    return df[['CAN ID','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8','Source','Target','label']]

def create_graphs_numpy(df, window_size=50, stride=50):
    """
    把 pandas DataFrame 切成 (window_size,stride) 的窗口，
    再用 window_data_transform_numpy 转成图
    """
    data_arr = df.to_numpy()  # shape: (N, 12) or so
    num_windows = (len(data_arr) - window_size) // stride + 1
    graphs = []
    for i in range(num_windows):
        start_idx = i * stride
        window = data_arr[start_idx : start_idx + window_size]
        g = window_data_transform_numpy(window)
        graphs.append(g)
    return graphs

def window_data_transform_numpy(data):

    source = data[:, 9]
    target = data[:, 10]
    labels = data[:, 11]

    # unique (source,target) + counts => edge_index, edge_attr
    unique_edges, edge_counts = np.unique(np.stack((source,target), axis=1), axis=0, return_counts=True)
    nodes = np.unique(np.concatenate((source, target)))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edge_index = np.vectorize(node_to_idx.get)(unique_edges).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_counts, dtype=torch.float).view(-1,1)

    node_features = np.zeros((len(nodes), 10), dtype=np.float32)
    for i, node in enumerate(nodes):
        # find rows where source==node
        mask = (source == node)
        # shape: (some_rows, 9) => columns [CAN ID..Data8] => index [0..8]
        sub_data = data[mask, 0:9]
        if len(sub_data) > 0:
            node_features[i, :9] = sub_data.mean(axis=0)
        node_features[i, 9] = mask.sum()

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([1 if 1 in labels else 0], dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return graph_data

def pad_row_vectorized(df):

    mask = df['DLC'] != 8
    for i in range(8, 1, -1):
        mask_label = mask & (df['DLC'] + 1 == i)
        df.loc[mask_label, 'label'] = df.loc[mask_label, f'Data{i}']
        df.loc[mask_label, f'Data{i}'] = '00'
    return df.fillna('00')

def hex_to_decimal_vectorized(series):
  
    return series.apply(lambda x: int(x,16) if pd.notnull(x) and x!='None' else None)

class GraphDataset(Dataset):
  
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
