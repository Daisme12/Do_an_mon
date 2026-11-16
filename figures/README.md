# Thư mục chứa hình ảnh (Figures)

Tất cả các hình ảnh, đồ thị, sơ đồ được sử dụng trong báo cáo đồ án nên được đặt tại đây:
- Định dạng khuyến nghị cho ảnh vector: `.pdf`, `.eps` (cho chất lượng in sắc nét tuyệt đối, không vỡ hạt).
- Định dạng cho ảnh chụp / ảnh raster: `.png`, `.jpg` (độ phân giải tối thiểu 300 DPI).
- Trong mã nguồn LaTeX, đường dẫn đã được định cấu hình tự động thông qua `\graphicspath{{figures/}}` trong file `config/preamble.tex`, do đó khi chèn ảnh bạn chỉ cần gọi tên file:
  ```latex
  \includegraphics[width=0.8\textwidth]{ten_hinh_anh.png}
  ```
