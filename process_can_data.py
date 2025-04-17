import os
import pandas as pd

import os
import pandas as pd

def unpack_data_field(df):
    df['data_field'] = df['data_field'].astype(str).str.strip()
    df['DLC'] = df['data_field'].apply(lambda x: len(x) // 2)
    df['bytes'] = df['data_field'].apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)])
    df['Label'] = df['attack'].map({0: 'R', 1: 'T'})
    return df

if __name__ == "__main__":
    input_file = "D:/quadeer/task1/dataprocessing/force-neutral-3.csv"
    df = pd.read_csv(input_file)
    df = unpack_data_field(df)

    output_file = os.path.join(os.path.dirname(__file__), "processed_force-neutral-3.csv")
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            base = [row['timestamp'], row['arbitration_id'], row['DLC']]
            bytes_part = row['bytes']
            label = row['Label']
            line_parts = base + bytes_part + [label]
            line = ",".join(map(str, line_parts))
            f.write(line + "\n")

    print(f"saved to: {output_file}")


