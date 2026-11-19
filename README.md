## 1. Phương pháp tính điểm Laplacian

Từ khung hình, crop lấy vùng bên trong bbox (ROI), rồi chuyển sang ảnh xám 8-bit. Với các bbox chồng nhau trong ảnh thử nghiệm, những vị trí bị viền hoặc nhãn che được bỏ khỏi phép tính điểm. Phép đo dùng kích thước pixel gốc (`1280x720`).

Laplacian đo mức thay đổi cường độ giữa mỗi pixel và bốn pixel lân cận. Bài thử dùng OpenCV với `CV_64F`, `ksize=1` và `BORDER_REFLECT_101`. Giá trị Laplacian có dấu được giữ nguyên để tính phương sai:

<img width="700" height="120" alt="image" src="https://github.com/user-attachments/assets/4f1eaa61-e423-4861-bb1b-77a86641d7f7" />

Công thức này lấy phương sai của các giá trị Laplacian để cho ra một điểm độ nét $S$ cho vùng ảnh xám bên trong bbox (ROI).

1. **Tính Laplacian tại từng pixel.** Với cấu hình đã thử, $L_i$ bằng tổng độ sáng của bốn pixel lân cận (trên, dưới, trái, phải) trừ bốn lần độ sáng của pixel ở giữa. Nếu pixel giữa có độ sáng $50$ và bốn pixel lân cận có độ sáng $65, 65, 50, 50$ thì:

<img width="352" height="27" alt="image" src="https://github.com/user-attachments/assets/b1d06896-0fe3-4257-a1c4-55e9f448040f" />

2. **Tính phương sai.** Lấy trung bình $\bar{L}$ của $N$ giá trị Laplacian hợp lệ. Với mỗi giá trị $L_i$, tính chênh lệch so với $\bar{L}$, bình phương rồi lấy trung bình. Bình phương giúp các giá trị âm và dương đều đóng góp vào điểm.

3. **Chạy với file smoke_laplacian.py**

Dùng Python 3.10 trở lên. Cài NumPy và bản OpenCV có hỗ trợ cửa sổ. Bản thử nghiệm đã chạy với OpenCV 5.0.0 và NumPy 2.2.6.

```powershell
python -m pip install opencv-python numpy
python smoke_laplacian.py "vid/true/media (9)_yolox_tiled.mp4" --output "demo_smoke.mp4" --show
```

Chạy từ thư mục chứa module. Giữ riêng `--output` để xuất MP4, giữ riêng `--show` để xem trực tiếp, hoặc dùng cả hai như ví dụ. Bấm **Q** hoặc **Esc** để dừng. Nếu dừng sớm, MP4 lưu phần đã chạy và kết quả JSON có `complete: false`, `decision: null`. Bản xuất ghi hình ảnh có đánh dấu ở FPS nguồn. Với clip có âm thanh, MP4 xuất từ module chỉ chứa luồng hình.

Ngưỡng mặc định là 300, yêu cầu 5 frame liên tiếp. Có thể thay bằng `--threshold` và `--min-consecutive`. Bbox `smoke` được ước lượng từ viền cam và vị trí nhãn confidence. Module cắt vào trong 4–5 px và bỏ vùng overlay chồng lên khi xác định được. ROI được chuyển xám rồi tính phương sai Laplacian có dấu với `ksize=1`. Frame qua ngưỡng nếu có ít nhất một bbox đạt điểm ≤ ngưỡng. Chuỗi được đếm theo frame của video, kể cả khi bbox qua ngưỡng thay đổi giữa các frame.

### Ví dụ minh họa

Xét ví dụ với ba giá trị Laplacian (cả hai vùng đều có $\bar{L} = 0$):

| Vùng ảnh | Giá trị $L_i$ | Điểm $S$ | Kết quả với ngưỡng 300 |
| --- | --- | --- | --- |
| Ít biến đổi độ sáng | $-3, 0, 3$ | $(9 + 0 + 9) / 3 = 6$ | Giữ |
| Biến đổi độ sáng mạnh | $-30, 0, 30$ | $(900 + 0 + 900) / 3 = 600$ | Loại |

Trong bài thử khói, điểm thấp cho thấy vùng ảnh ít có biến đổi độ sáng đột ngột. Cả vùng khói mờ và vùng phản sáng trơn đều có thể cho điểm thấp, nên bbox được giữ ở bước này vẫn cần kiểm tra tiếp.

## 2. Kết quả trên ảnh và video

| Dữ liệu | Thư mục | Tổng | Giữ | Loại |
| --- | --- | --- | --- | --- |
| Video | `false` | 7 | 2 | 5 |
| Video | `true` | 11 | 9 | 2 |

- **Video:** Video được giữ nếu có ít nhất **5 frame gốc liên tiếp**, mỗi frame có ít nhất một bbox `smoke` đọc được và có phương sai Laplacian $\le 300$. Frame thiếu bbox qua ngưỡng làm đứt chuỗi. Quy tắc áp dụng ở cấp frame, không ghép ID bbox giữa các frame. Bộ lọc loại **5/7 mẫu `false` (71.4%)** và giữ **9/11 mẫu `true` (81.8%)**.

## 3. Kết luận, ưu điểm và hạn chế

**Kết luận:** Kết quả ủng hộ việc dùng điểm Laplacian như một tín hiệu phụ. Ở ngưỡng `300`, phép thử trên ảnh giữ 11/20 mẫu `true`, còn video giữ 9/11 mẫu `true`. Vẫn cần nghiên cứu thêm và test để tìm ra các ngưỡng phù hợp trước khi dùng điểm để loại cảnh báo tự động.

- **Ưu điểm:**
  - Phép tính đơn giản, nhanh, dễ kiểm tra theo từng bbox và có thể bổ sung sau YOLOX mà không cần huấn luyện lại.
  - Lọc tốt với các case áo mưa, áo khoác nhầm thành khói.
- **Hạn chế:** Phương sai Laplacian đo thay đổi cường độ, nên viền xe, vạch sàn, nhiễu, nén video và mép bbox đều có thể ảnh hưởng. Bề mặt phản sáng hoặc áo trơn có thể cho điểm thấp. Khói nằm trước nền nhiều chi tiết có thể cho điểm cao.
- **Hướng kiểm chứng:** Lấy ROI trực tiếp từ khung hình trước khi vẽ bbox, thử tổng hợp theo nhiều khung của cùng một vùng theo dõi, rồi đánh giá tỷ lệ bỏ sót mẫu `true` và tỷ lệ loại mẫu `false` trên một tập video có nhãn riêng.
