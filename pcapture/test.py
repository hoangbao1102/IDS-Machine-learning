# import pandas as pd

# def check_header_mismatch(file1_path, file2_path):
#     # Read headers of both CSV files
#     header1 = pd.read_csv(file1_path, nrows=0).columns.tolist()
#     header2 = pd.read_csv(file2_path, nrows=0).columns.tolist()

#     # Normalize the headers by stripping whitespace and converting to lowercase
#     header1 = [col.strip().lower() for col in header1]
#     header2 = [col.strip().lower() for col in header2]

#     # Iterate through headers and check for mismatches
#     for column1, column2 in zip(header1, header2):
#         if column1 == column2:
#             print(column1, ' ', column2, '------------------------Equal')
#         else:
#             print(column1, ' ', column2, '-----------Not Equal')

# # Example usage:
# file1_path = "/home/kali/Cuoi-Ky-ATTT/IDS-Machine-learning/packet_features.csv"
# file2_path = "/home/kali/Cuoi-Ky-ATTT/IDS-Machine-learning/pcapture/ref.csv"
# check_header_mismatch(file1_path, file2_path)
import pandas as pd

def compare_csv_columns(file1_path, file2_path):
    # Đọc các file CSV
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Lấy danh sách các cột và chuyển thành chữ thường
    columns_file1 = [col.lower() for col in df1.columns]
    columns_file2 = [col.lower() for col in df2.columns]

    # In ra tên các cột theo thứ tự
    max_length = max(len(columns_file1), len(columns_file2))
    for i in range(max_length):
        if i < len(columns_file1):
            print(f"Cột {i + 1} file 1: {columns_file1[i]}")
        if i < len(columns_file2):
            print(f"Cột {i + 1} file 2: {columns_file2[i]}")

# Ví dụ sử dụng
file1_path = "/home/kali/Cuoi-Ky-ATTT/IDS-Machine-learning/packet_features.csv"
file2_path = "/home/kali/Cuoi-Ky-ATTT/IDS-Machine-learning/pcapture/ref.csv"
compare_csv_columns(file1_path, file2_path)
