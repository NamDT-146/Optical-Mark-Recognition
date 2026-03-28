### 📅 Kế hoạch triển khai Dự án: Ứng dụng Chấm điểm Trắc nghiệm Tự động

| Checkpoint | Tuần (Ngày) | Giai đoạn | Mô tả Công việc Chi tiết | Tiêu chí Hoàn thành (DoD) |
| :--- | :--- | :--- | :--- | :--- |
| **CP 1** | **29** (30/03) | **Lập kế hoạch & Pipeline cơ bản** | • Lập kế hoạch kiến trúc hệ thống (Sơ đồ khối).<br>• Cài đặt môi trường phát triển (OpenCV, CMake).<br>• Xây dựng module tạo PDF form & parser Excel đáp án.<br>• Xây dựng pipeline xử lý ảnh thô (Nắn ảnh & tách vùng ROI). | • Tài liệu kiến trúc được chốt.<br>• Module PDF/Excel hoạt động độc lập.<br>• Thuật toán nắn ảnh chạy được trên console. |
| **CP 2** | **31** (13/04) | **Hoàn thiện Core OMR C++** | • Cài đặt thuật toán nhận diện hình dạng (Contour) & ô tô đậm.<br>• Xây dựng logic chấm điểm & so khớp đáp án.<br>• Hoàn thiện module xuất báo cáo CSV/Text.<br>• Kiểm thử độ chính xác trên bộ ảnh mẫu (Testset). | • Pipeline C++ thông suốt: Ảnh + Đáp án -\> Kết quả.<br>• Độ chính xác nhận diện ban đầu đạt **\>80%**. |
| **CP 3** | **33** (27/04) | **MVP PC & Tích hợp Android NDK** | • Xây dựng **Localhost Web UI** (FastAPI) để test luồng thực tế.<br>• Thiết kế UI/UX cho ứng dụng Mobile (Figma).<br>• Cấu hình Android NDK & viết JNI Wrapper.<br>• Đưa Core C++ xuống môi trường di động. | • Web Demo hoạt động hoàn chỉnh.<br>• App Android gọi được hàm xử lý C++ qua JNI thành công. |
| **CP 4** | **35** (11/05) | **CameraX & Real-time Data** | • Tích hợp CameraX, truyền frame trực tiếp qua JNI.<br>• Xử lý đa luồng (Background Thread) để tránh treo UI.<br>• Tích hợp SQLite/Room để lưu lịch sử chấm bài. | • Quét và hiển thị kết quả từ camera với **FPS \> 1**.<br>• Người dùng xem được danh sách bài đã chấm trên App. |
| **CP 5** | **37** (25/05) | **Tối ưu & Đóng gói APK** | • Hoàn thiện tính năng hỗ trợ: Chỉnh sửa đáp án, biểu điểm...<br>• Xây dựng tính năng chia sẻ báo cáo (PDF/CSV) qua Google Drive.<br>• Tối ưu hiệu năng, sửa lỗi crash khi chấm số lượng lớn. | • Giao diện mượt mà, tính năng chia sẻ thành công.<br>• Build thành công file **APK bản Release**. |
| **CP 6** | **39+** (TBD) | **Báo cáo & Bảo vệ** | • Viết tài liệu thuật toán và thử nghiệm thực tế.<br>• Thu thập kết quả chạy thực tế với bài thi thật.<br>• Chuẩn bị slide bảo vệ và nộp Báo cáo tổng kết. | • Không còn lỗi nghiêm trọng (Critical bugs).<br>• Báo cáo chi tiết, source code hoàn thiện có hướng dẫn. |

-----

### 📂 Cấu trúc Codebase (Dự kiến cho Github)

```text
Project2_AutoGrader/
├── core/               # Core xử lý C++ & OpenCV
├── services/           # Python services (Generate PDF, Parse Excel)
├── ui/                 # Localhost Web Demo (FastAPI)
├── test/               # Bộ dữ liệu mẫu để test (Images, CSV)
└── docs/               # Sơ đồ kiến trúc & Báo cáo tuần
```