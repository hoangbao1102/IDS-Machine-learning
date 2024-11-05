import pandas as pd

def check_header_mismatch(file1_path, file2_path):
    # Read headers of both CSV files
    header1 = pd.read_csv(file1_path, nrows=0).columns.tolist()
    header2 = pd.read_csv(file2_path, nrows=0).columns.tolist()

    # Normalize the headers by stripping whitespace and converting to lowercase
    header1 = [col.strip().lower() for col in header1]
    header2 = [col.strip().lower() for col in header2]

    # Iterate through headers and check for mismatches
    for column1, column2 in zip(header1, header2):
        if column1 == column2:
            print(column1, ' ', column2, '------------------------Equal')
        else:
            print(column1, ' ', column2, '-----------Not Equal')

# Example usage:
file1_path = "C:/DUT/Nam5/Ky9/ATTT/Cuoi Ky/packet_features.csv"
file2_path = "C:/DUT/Nam5/Ky9/ATTT/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
check_header_mismatch(file1_path, file2_path)
