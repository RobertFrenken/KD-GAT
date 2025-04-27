import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import os
import unittest

def graph_creation(root_folder, folder_type='train_', window_size=50, stride=50):
    """
    Creates a dataset of graphs from a set of CSV files containing CAN data.
    
    Args:
        combined (bool): If True, combines multiple datasets into one.
        path (string): A string containing the path to a CAN data csv file.
        datasize (float): The size of the dataset to be used. Default is 1.0 (100%).
        window_size (int): The size of the sliding window.
        stride (int): The stride for the sliding window.
    
    Returns:
        dataset: Class object GraphDataset pytorch geometric graph datasets.
    """
    # Find all CSV files in folders with 'train' in their name
    train_csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if folder_type in dirpath.lower():  # Check if 'train_' is in the folder name
            for filename in filenames:
                if filename.endswith('.csv'):  # Only include CSV files
                    train_csv_files.append(os.path.join(dirpath, filename))
    
    # Process each CSV file and create graphs
    combined_list = []
    for csv_file in train_csv_files:
        print(f"Processing file: {csv_file}")
        df = dataset_creation_vectorized(csv_file)
        # Check for NaN values in the DataFrame
        # this is a stopgap. The issue is there are some rows where
        # the data field is empty, and the column Data1 is left Nan
        # in the future will need to handle this in the above function.
        if df.isnull().values.any():
            print(f"NaN values found in DataFrame from file: {csv_file}")
            print(df[df.isnull().any(axis=1)])
            df.fillna(0, inplace=True)  # Replace NaN values with 0
        graphs = create_graphs_numpy(df, window_size=window_size, stride=stride)
        combined_list.extend(graphs)

    # Create the combined dataset
    dataset = GraphDataset(combined_list)
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
    start_indices = range(0, num_windows * stride, stride)

    # list comprehension to create graphs for each window
    return [window_data_transform_numpy(data[start:start + window_size]) 
            for start in start_indices]


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
    target = data[:, -2]  # Assuming Target is the second to last column
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
    
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([1 if 1 in labels else 0], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def pad_row_vectorized(df):
    """
    Pads rows in a Pandas DataFrame where the data length code (DLC) is less than 8
    by filling missing columns with a hex code value of '00'.

    Args:
        df (pandas.DataFrame): A pandas DataFrame.

    Returns:
        df: A pandas DataFrame with missing columns padded to '00'.
    """
    # Create a mask for rows where DLC is less than 8
    mask = df['DLC'] < 8

    # Iterate over the range of Data1 to Data8 columns
    for i in range(8):
        # Only pad columns where the index is greater than or equal to the DLC
        column_name = f'Data{i+1}'
        df.loc[mask & (df['DLC'] <= i), column_name] = '00'

    # Fill any remaining NaN values with '00'
    df.fillna('00', inplace=True)

    return df
def dataset_creation_vectorized(path):
    """
    Takes a csv file containing CAN data. Creates a pandas dataframe,
    adds source and target columns, pads the rows with missing values,
    transforms the hex values to decimal values, and reencodes the labels
    to a binary classifcation problem of 0 for attack free and 1 for attack.
    
    Args:
        path (string): A string containing the path to a CAN data csv file.
    
    Returns:
        df: a pandas dataframe.
    """
    df = pd.read_csv(path)
    df.columns = ['Timestamp', 'arbitration_id', 'data_field', 'attack']
    df.rename(columns={'arbitration_id': 'CAN ID'}, inplace=True)

    # Ensure 'data_field' is a string and handle missing values
    df['data_field'] = df['data_field'].astype(str).fillna('')

    # Add the DLC column based on the length of the data_field
    df['DLC'] = df['data_field'].apply(lambda x: len(x) // 2)

    # Unpack the data_field column into individual bytes
    df['data_field'] = df['data_field'].astype(str).str.strip()  # Ensure it's a string and strip whitespace
    df['bytes'] = df['data_field'].apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)])  # Split into bytes

    # Determine the maximum number of bytes (should be 8 or fewer)
    max_bytes = 8
    # Create Data1 to Data8 columns, padding with '00' if fewer than 8 bytes
    for i in range(max_bytes):
        df[f'Data{i+1}'] = df['bytes'].apply(lambda x: x[i] if i < len(x) else '00')

    df['Source'] = df['CAN ID']
    df['Target'] = df['CAN ID'].shift(-1)

    # Pad rows and fill missing values
    df = pad_row_vectorized(df)

    # Convert hex columns to decimal
    hex_columns = ['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 
                   'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target']
    # Convert hex values to decimal
    for col in hex_columns:
        df[col] = df[col].apply(lambda x: int(x, 16) if pd.notnull(x) and isinstance(x, str) and all(c in '0123456789abcdefABCDEF' for c in x) else None)

    # Drop the last row and reencode labels
    df = df.iloc[:-1]
    
    # Map the attack column directly to the label column
    df['label'] = df['attack'].astype(int)

    return df[['CAN ID', 'Data1', 'Data2', 'Data3', 'Data4', 
               'Data5', 'Data6', 'Data7', 'Data8', 'Source', 'Target', 'label']]

class GraphDataset(Dataset):
    """
    Takes a list of pytorch geometric Data objects and creates a dataset.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

#################################
# Testing Class                 #
#################################
class TestPreprocessing(unittest.TestCase):
    def test_dataset_creation_vectorized(self):
        test_path = r"datasets/can-train-and-test-v1.5/hcrl-ch/train_02_with_attacks/fuzzing-train.csv"
        df = dataset_creation_vectorized(test_path)
        self.assertEqual(len(df.columns), 12)
        self.assertTrue('label' in df.columns)

        # Check for NaN values in the DataFrame
        self.assertFalse(df.isnull().values.any(), "Dataset contains NaN values!")

    def test_graph_creation(self):
        root_folder = r"datasets/can-train-and-test-v1.5/set_02"
        graph_dataset = graph_creation(root_folder)
        self.assertGreater(len(graph_dataset), 0)

         # Check for NaN values in the graph dataset
        for data in graph_dataset:
            self.assertFalse(torch.isnan(data.x).any(), "Graph dataset contains NaN values!")
            self.assertFalse(torch.isinf(data.x).any(), "Graph dataset contains Inf values!")


if __name__ == "__main__":
    unittest.main()