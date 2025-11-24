🌸 Bài toán phân loại hoa – Oxford Flowers 102

Trong lĩnh vực thị giác máy tính, việc nhận diện chính xác các loài hoa từ hình ảnh là một bài toán quan trọng, có ứng dụng trong nông nghiệp, chăm sóc cây cảnh, nghiên cứu sinh học và xây dựng các hệ thống nhận dạng tự động. Bộ dữ liệu Oxford Flowers 102 gồm 8.189 hình ảnh thuộc 102 loài hoa khác nhau, với độ khó cao do hình dạng và màu sắc các loài rất giống nhau.

🎯 Mục đích nghiên cứu

Xây dựng mô hình có khả năng phân loại chính xác hình ảnh hoa thuộc 102 lớp.

So sánh hiệu quả giữa hai phương pháp:

HOG + SVM (Machine Learning truyền thống)

ResNet50 Feature Extraction + Classifier (Deep Learning)

Tìm ra phương pháp phù hợp hơn cho bài toán nhận dạng hình ảnh tự nhiên.

Xây dựng một ứng dụng web giúp tải ảnh lên và dự đoán loài hoa nhanh chóng.

🤖 Thuật toán áp dụng
1. HOG + SVM (Histogram of Oriented Gradients + Support Vector Machine)

Phương pháp này sử dụng đặc trưng thủ công:

Ảnh được chuyển sang grayscale và resize về 128×128.

HOG trích xuất đặc trưng dựa trên gradient, biên dạng và hướng cạnh.

SVM kernel RBF phân loại dựa trên vector đặc trưng.

Ưu điểm: nhanh, đơn giản, dễ chạy trên máy yếu.
Nhược điểm: độ chính xác không cao với dữ liệu phức tạp như ảnh hoa.

2. ResNet50 Feature Extraction + Fully Connected Classifier

Phương pháp Deep Learning:

Dùng ResNet50 pretrained trên ImageNet để trích xuất vector đặc trưng 2048 chiều từ ảnh 224×224.

Xây dựng mạng phân loại gồm:

Dense 512 (ReLU)

Dropout 0.5

Dense 102 (Softmax)

Ưu điểm: độ chính xác cao, nhận diện tốt ngay cả khi hoa có hình dạng gần giống nhau.
Nhược điểm: thời gian train lâu hơn, yêu cầu GPU để đạt hiệu suất tốt.

🔧 Công cụ và thư viện sử dụng

Ngôn ngữ: Python

Thư viện chính:

numpy, pandas, matplotlib, Pillow

scikit-learn cho HOG, SVM, train/test split

tensorflow / keras cho ResNet50 và mô hình phân loại

tqdm cho progress bar

Flask để triển khai giao diện web

Giao diện người dùng: Bootstrap 5 (form upload + hiển thị kết quả)

📌 Quy trình xử lý dữ liệu

Khám phá dữ liệu (EDA):

Đếm số ảnh mỗi lớp

Hiển thị ảnh mẫu của các lớp

Kiểm tra chất lượng và sự phân bố ảnh

Tiền xử lý ảnh:

Resize thống nhất: 128×128 (HOG), 224×224 (ResNet50)

Chuẩn hóa pixel

Chuyển grayscale (với HOG)

Tiền xử lý ImageNet (với ResNet)

Trích xuất đặc trưng:

HOG vector (≈ 4700 chiều)

ResNet50 feature (2048 chiều)

Huấn luyện mô hình:

SVM RBF (phân loại HOG)

Fully Connected Network (phân loại ResNet features)

Đánh giá mô hình:

Classification Report

Accuracy trên tập test

So sánh HOG vs ResNet

Triển khai web:

Tải ảnh → xử lý → dự đoán → trả kết quả Top-1 & Top-5

🌼 Kết quả kỳ vọng

Hệ thống dự kiến phân loại hoa thành các loài rõ ràng, ví dụ:

Các nhóm hoa có màu sắc tương tự (hoa tím, hoa vàng, hoa đỏ).

Các loài có dạng cánh tròn, cánh dài, cánh chùm.

Các loài tương đồng về cấu trúc hình dạng nhưng khác sắc thái màu.

Thông thường:

HOG + SVM cho accuracy trung bình → phù hợp bài thực hành ML cơ bản.

ResNet50 đạt accuracy cao → phù hợp bài toán thực tế.

<img width="604" height="202" alt="{F33D9FA0-BC74-4BF6-B666-D3BA79DEDBFF}" src="https://github.com/user-attachments/assets/7b1755cd-c5c1-4cb1-bf52-dcbfcb8f7bab" />

kết quả

<img width="676" height="477" alt="{C5B50E32-981C-4D67-A55C-737A4BD5C524}" src="https://github.com/user-attachments/assets/fac5b19e-e9b3-4934-871b-1d5602f4789a" />

