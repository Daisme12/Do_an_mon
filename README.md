# ĐỒ ÁN MÔN HỌC — ĐỀ TÀI 5
## Khôi phục tín hiệu thưa bằng cực tiểu hóa chuẩn $\ell_1$

- **Đơn vị:** Trường Đại học Giao thông Vận tải (UTC) — Khoa Công nghệ thông tin — Lớp: Toán ứng dụng
- **Giảng viên hướng dẫn:** Nguyễn Thế Vinh
- **Tài liệu biên dịch hoàn chỉnh:** [`main.pdf`](main.pdf) (57 trang)

---

## 1. Tóm tắt nhanh nội dung các chương

Tài liệu được thiết kế theo mạch logic từ **bài toán thực tế $\rightarrow$ mô hình tối ưu $\rightarrow$ điều kiện đảm bảo $\rightarrow$ thuật toán giải số $\rightarrow$ thực nghiệm & tự kiểm tra**:

### 📖 Chương 1: Tổng quan và cơ sở về tín hiệu thưa (`chapters/chuong1_tong_quan.tex`)
- **Vấn đề đặt ra:** Vượt qua giới hạn lấy mẫu cổ điển Shannon--Nyquist bằng hệ phương trình thiếu ẩn $Ax = b$ ($A \in \mathbb{R}^{m \times n}, m < n$).
- **Biểu diễn thưa:** Tín hiệu tự nhiên thưa trong một cơ sở/từ điển $\Psi$ (Fourier, Wavelet): $x = \Psi\theta \implies b = A\Psi\theta + e$.
- **Công cụ đo độ thưa:** Phân biệt giả chuẩn $\ell_0$, chuẩn $\ell_1$ và chuẩn $\ell_2$ (**Bảng 1.1**), chứng minh bất đẳng thức kép $\|x\|_2 \le \|x\|_1 \le \sqrt{n}\|x\|_2$.
- **Tín hiệu thưa xấp xỉ:** Định nghĩa và chứng minh bài toán sai số xấp xỉ $k$-thưa tốt nhất $\sigma_k(x)_1$.

### 📐 Chương 2: Các mô hình tối ưu và hình học của chuẩn $\ell_1$ (`chapters/chuong2_mo_hinh_toi_uu.tex`)
- **Hệ thống 4 mô hình tối ưu (**Bảng 2.1**):** $(P_0)$ (NP-hard) $\rightarrow$ giải tỏa lồi sang **Basis Pursuit (BP)** (không nhiễu) $\rightarrow$ **BPDN** và **LASSO** (có nhiễu).
- **Trực giác hình học (**Hình 2.1**):** Lý giải qua không gian $\mathbb{R}^2$ tại sao quả cầu $\ell_1$ (hình thoi có đỉnh nhọn trên trục) tạo ra nghiệm thưa, còn quả cầu $\ell_2$ (hình tròn trơn) tạo ra nghiệm đặc.
- **Giải tích lồi & Dưới vi phân:** Tính dưới vi phân $\partial |t|$ và $\partial \|x\|_1$, thiết lập điều kiện KKT cho LASSO ($|a_i^T r^*| < \lambda \implies x_i^* = 0$) và chứng minh công thức tường minh cho **toán tử co ngưỡng mềm** $\mathcal{S}_t(v)$ (**Hình 2.2**).

### 🛡️ Chương 3: Các điều kiện khôi phục và tính ổn định (`chapters/chuong3_dieu_kien_khoi_phuc.tex`)
Trả lời trọn vẹn **3 câu hỏi lý thuyết cốt lõi** của Lấy mẫu nén:
1. **Spark ($\text{spark}(A) > 2k$):** Điều kiện cần và đủ cho **tính duy nhất** của nghiệm $k$-thưa (đòi hỏi $m \ge 2k$).
2. **NSP (Null Space Property):** Điều kiện cần và đủ để **thuật toán lồi Basis Pursuit** tìm đúng nghiệm gốc.
3. **Mutual Coherence ($\mu(A)$):** Chỉ số đại số tính nhanh $\mathcal{O}(mn^2)$ cho điều kiện đủ $k < \frac{1}{2}(1 + 1/\mu(A))$.
4. **RIP (Restricted Isometry Property):** Điều kiện $\delta_{2k} < \sqrt{2}-1$ bảo đảm **khôi phục ổn định khi có nhiễu** ($\|\widehat{x} - x^\natural\|_2 \le C_0 \frac{\sigma_k(x^\natural)_1}{\sqrt{k}} + C_1 \varepsilon$).
- **Tổng hợp đối sánh:** **Bảng 3.1** so sánh cả 4 tiêu chuẩn và **Hình 3.1** minh họa sơ đồ phân tầng kéo theo logic giữa chúng.

### ⚙️ Chương 4: Thuật toán giải số và phân tích hội tụ (`chapters/chuong4_thuat_toan.tex`)
Xây dựng bước lặp và phân tích hội tụ cho 4 thuật toán tiêu biểu (**Bảng 4.1**):
- **ISTA:** Hạ bậc hai kết hợp co ngưỡng mềm; chứng minh chi tiết bất đẳng thức 3 điểm và tốc độ hội tụ $\mathcal{O}(1/k)$.
- **FISTA:** Áp dụng gia tốc động lượng Nesterov đạt tốc độ tối ưu $\mathcal{O}(1/k^2)$.
- **GPSR:** Tách biến $x = u - v$ ($u, v \ge 0$), chiếu gradient kết hợp bước Barzilai--Borwein.
- **ADMM (**Hình 4.1**):** Phân tách biến giải chính xác bài toán Basis Pursuit bằng 3 bước luân phiên (chiếu trực giao lên $Ax=b$, co ngưỡng mềm, cập nhật đối ngẫu).

### 📊 Chương 5: Thực nghiệm mô phỏng và câu hỏi tự kiểm tra (`chapters/chuong5_thuc_nghiem.tex`)
- **5 kịch bản thực nghiệm số:**
  1. Đường cong chuyển pha Donoho--Tanner (**Hình 5.1**).
  2. Ảnh hưởng của nhiễu (SNR) và tinh chỉnh $\lambda$ trong LASSO (**Hình 5.2**).
  3. So sánh tốc độ hội tụ thực tế ISTA vs FISTA theo thang logarit (**Hình 5.3**).
  4. Tác động của Mutual Coherence cao ($\mu \approx 1$) làm sụt giảm tỷ lệ khôi phục.
  5. Nén và tái tạo ảnh xám 2D dùng biến đổi Wavelet db4 (**Hình 5.4**).
- **8 câu hỏi tự kiểm tra & lời giải chi tiết:** Củng cố từ lý thuyết lồi, Spark/NSP/RIP đến cơ chế KKT và đánh giá CPU time.
- **Khung tiêu chí đánh giá đồ án (**Bảng 5.1**).**

### 🎯 Kết luận và hướng phát triển (`chapters/ket_luan.tex`)
Tổng kết lời giải cho 3 câu hỏi nền tảng và gợi mở các hướng nghiên cứu tiếp theo: *Total Variation (TV)*, *Group Sparsity*, *Matrix Completion* và *Deep Unfolding (LISTA)*.

---

## 2. Cấu trúc thư mục mã nguồn

```text
Do_an_mon/
├── main.tex                  # File LaTeX gốc (compile từ file này)
├── main.pdf                  # File báo cáo PDF đã biên dịch (57 trang)
├── config/
│   ├── preamble.tex          # Cấu hình gói lệnh, lề trang, font Times New Roman
│   └── commands.tex          # Định nghĩa các ký hiệu toán học (\R, \normone, \spark,...)
├── frontmatter/
│   ├── cover.tex             # Trang bìa chuẩn UTC
│   └── abstract.tex          # Lời mở đầu
├── chapters/
│   ├── chuong1_tong_quan.tex
│   ├── chuong2_mo_hinh_toi_uu.tex
│   ├── chuong3_dieu_kien_khoi_phuc.tex
│   ├── chuong4_thuat_toan.tex
│   ├── chuong5_thuc_nghiem.tex
│   └── ket_luan.tex
├── figures/                  # Ảnh logo UTC và tài nguyên hình ảnh
└── bibliography/
    └── references.bib        # Danh mục tài liệu tham khảo BibTeX
```

---

## 3. Hướng dẫn biên dịch LaTeX (macOS / VS Code)

Dự án sử dụng **XeLaTeX** (hỗ trợ trực tiếp tiếng Việt Unicode và font hệ thống `Times New Roman`).

- **Biên dịch bằng dòng lệnh:**
  ```bash
  /Library/TeX/texbin/xelatex -interaction=nonstopmode main.tex
  bibtex main
  /Library/TeX/texbin/xelatex -interaction=nonstopmode main.tex
  /Library/TeX/texbin/xelatex -interaction=nonstopmode main.tex
  ```
- **Biên dịch trên VS Code:** Mở file `main.tex` và nhấn nút **Build LaTeX project** (đã cấu hình sẵn recipe `XeLaTeX` trong `.vscode/settings.json`).
