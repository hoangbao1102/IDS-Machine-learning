Tất cả chương trình được chứa trong folder "Chuong trinh", trong đó:
- sniff: chứa chương trình chính
- pcapture: chứa các thử nghiệm bắt gói tin
- test: chứa các xử lý nghiệp vụ
- trained: chứa các data mẫu và các model đã được huấn luyện cũng như code huấn luyện 

Để chạy chương trình: vào sniff/sniff_packet.py
- chỉnh sửa interface bạn muốn bắt gói tin
- chỉnh sửa model và thư viện tương ứng để load model
- chạy python sniff_packet.py
-> kết quả sẽ được in ra terminal, các gói tin bắt được ở mỗi vòng lặp được lưu vào packet_features.
