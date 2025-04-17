import os
import pandas as pd

def unpack_data_field(df):
    df['data_field'] = df['data_field'].astype(str).str.strip()
    df['DLC'] = df['data_field'].apply(lambda x: len(x) // 2)
    df['bytes'] = df['data_field'].apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)])

    max_bytes = df['bytes'].apply(len).max()
    for i in range(max_bytes):
        df[f'Data{i}'] = df['bytes'].apply(lambda x: x[i] if i < len(x) else None)

    df['Label'] = df['attack'].map({0: 'R', 1: 'T'})
    data_cols = [f'Data{i}' for i in range(max_bytes)]
    return df[['timestamp', 'arbitration_id', 'DLC'] + data_cols + ['Label']]

def process_csv_directory(input_root, output_root):
    for dirpath, _, filenames in os.walk(input_root):
        relative_path = os.path.relpath(dirpath, input_root)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        for filename in filenames:
            if filename.endswith('.csv'):
                input_csv_path = os.path.join(dirpath, filename)
                output_csv_path = os.path.join(output_dir, f"processed_{filename}")

                try:
                    df = pd.read_csv(input_csv_path)
                    df_processed = unpack_data_field(df)
                    df_processed.to_csv(output_csv_path, index=False, header=False, na_rep='')
                    print(f"Processed: {output_csv_path}")
                except Exception as e:
                    print(f"Failed to process {input_csv_path}: {e}")

if __name__ == "__main__":
    input_root = r"D:/quadeer/task1/dataprocessing/brooke-lampe-can-train-and-test-2d293b6d0439"
    output_root = r"D:/quadeer/task1/dataprocessing/processed_brooke_lampe"
    process_csv_directory(input_root, output_root)
