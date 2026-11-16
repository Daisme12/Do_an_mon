"""
Mã nguồn thực nghiệm mô phỏng cho Đồ án Môn học (Đề tài 5):
"Khôi phục tín hiệu thưa bằng cực tiểu hóa chuẩn l1"

Môi trường chạy: Python 3 + NumPy + Matplotlib + PyWavelets
Seed cố định: 42 (đảm bảo khả năng tái lập hoàn toàn kết quả số).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pywt

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# CÁC HÀM CỐT LÕI: TOÁN TỬ CO NGƯỠNG MỀM & CÁC THUẬT TOÁN TỐI ƯU
# ==============================================================================

def soft_threshold(v: np.ndarray, tau: float) -> np.ndarray:
    """Toán tử co ngưỡng mềm S_tau(v) = sign(v) * max(|v| - tau, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def power_iteration_lipschitz(A: np.ndarray, n_iters: int = 30) -> float:
    """Ước lượng hằng số Lipschitz L = ||A||_2^2 = lambda_max(A^T A)."""
    v = np.ones(A.shape[1]) / np.sqrt(A.shape[1])
    for _ in range(n_iters):
        Av = A @ v
        v_next = A.T @ Av
        norm = np.linalg.norm(v_next)
        if norm == 0:
            return 1.0
        v = v_next / norm
    return float(np.linalg.norm(A @ v) ** 2) * 1.02  # hệ số an toàn nhỏ


def lasso_obj(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    r = A @ x - b
    return 0.5 * float(r @ r) + lam * float(np.linalg.norm(x, 1))


def ista_lasso(A: np.ndarray, b: np.ndarray, lam: float, L: float,
               max_iter: int = 400, tol: float = 1e-8):
    """Thuật toán ISTA giải bài toán LASSO."""
    t0 = time.perf_counter()
    x = np.zeros(A.shape[1])
    F_hist = [lasso_obj(A, b, x, lam)]
    step = 1.0 / L
    tau = lam * step
    for _ in range(max_iter):
        grad = A.T @ (A @ x - b)
        x_next = soft_threshold(x - step * grad, tau)
        F_hist.append(lasso_obj(A, b, x_next, lam))
        if np.linalg.norm(x_next - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_next
            break
        x = x_next
    return x, np.array(F_hist), time.perf_counter() - t0


def fista_lasso(A: np.ndarray, b: np.ndarray, lam: float, L: float,
                max_iter: int = 400, tol: float = 1e-8):
    """Thuật toán FISTA (gia tốc Nesterov) giải bài toán LASSO."""
    t0 = time.perf_counter()
    x = np.zeros(A.shape[1])
    y = x.copy()
    t = 1.0
    F_hist = [lasso_obj(A, b, x, lam)]
    step = 1.0 / L
    tau = lam * step
    for _ in range(max_iter):
        grad = A.T @ (A @ y - b)
        x_next = soft_threshold(y - step * grad, tau)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_next + ((t - 1.0) / t_next) * (x_next - x)
        F_hist.append(lasso_obj(A, b, x_next, lam))
        if np.linalg.norm(x_next - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_next
            break
        x = x_next
        t = t_next
    return x, np.array(F_hist), time.perf_counter() - t0


def admm_bp(A: np.ndarray, b: np.ndarray, rho: float = 1.0,
            max_iter: int = 600, tol: float = 1e-5):
    """Thuật toán ADMM giải bài toán Basis Pursuit: min ||x||_1 s.t. Ax = b."""
    m, n = A.shape
    AAt = A @ A.T
    L_chol = np.linalg.cholesky(AAt + 1e-12 * np.eye(m))

    def solve_AAt(rhs):
        y = np.linalg.solve(L_chol, rhs)
        return np.linalg.solve(L_chol.T, y)

    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    for _ in range(max_iter):
        v = z - u
        x = v - A.T @ solve_AAt(A @ v - b)
        z_prev = z
        z = soft_threshold(x + u, 1.0 / rho)
        u = u + x - z
        if np.linalg.norm(x - z) < tol and rho * np.linalg.norm(z - z_prev) < tol:
            break
    return z


# ==============================================================================
# THÍ NGHIỆM 1: KHẢO SÁT ĐƯỜNG CHUYỂN PHA DONOHO--TANNER
# ==============================================================================

def run_exp1_phase_transition():
    print("\n=== THÍ NGHIỆM 1: CHUYỂN PHA BASIS PURSUIT (ADMM) ===")
    n = 128
    m_vals = [20, 35, 50, 65, 80, 95]
    k_vals = [4, 8, 12, 16, 20, 24, 28, 32]
    T = 15
    prob = np.zeros((len(k_vals), len(m_vals)))

    for j, m in enumerate(m_vals):
        for i, k in enumerate(k_vals):
            if k >= m:
                prob[i, j] = 0.0
                continue
            success = 0
            for _ in range(T):
                A = rng.standard_normal((m, n)) / np.sqrt(m)
                A /= np.linalg.norm(A, axis=0, keepdims=True)
                x_true = np.zeros(n)
                supp = rng.choice(n, size=k, replace=False)
                x_true[supp] = rng.standard_normal(k)
                b = A @ x_true
                x_hat = admm_bp(A, b, rho=1.0, max_iter=500, tol=1e-4)
                rel_err = np.linalg.norm(x_hat - x_true) / max(1.0, np.linalg.norm(x_true))
                if rel_err < 1e-3:
                    success += 1
            prob[i, j] = success / T

    print("Bảng tỷ lệ thành công P(m, k) (hàng: k, cột: m):")
    header = "k \\ m | " + " | ".join(f"{m:4d}" for m in m_vals)
    print(header)
    print("-" * len(header))
    for i, k in enumerate(k_vals):
        row = f"{k:5d} | " + " | ".join(f"{prob[i, j]:4.2f}" for j in range(len(m_vals)))
        print(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(prob, origin="lower", aspect="auto", cmap="RdYlBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(m_vals)))
    ax.set_xticklabels([f"{m}\n({m/n:.2f})" for m in m_vals], fontsize=9)
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels(k_vals, fontsize=9)
    ax.set_xlabel(r"Số phép đo $m$ (và tỷ lệ nén $\delta = m/n$)", fontsize=10)
    ax.set_ylabel(r"Độ thưa $k$", fontsize=10)
    ax.set_title(r"Xác suất khôi phục thành công Basis Pursuit ($n = 128, T = 15$)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tỷ lệ thành công", fontsize=9)
    for i in range(len(k_vals)):
        for j in range(len(m_vals)):
            ax.text(j, i, f"{prob[i, j]:.2f}", ha="center", va="center",
                    color="black" if 0.25 < prob[i, j] < 0.85 else "white", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_exp1_phase_transition.pdf"))
    plt.close(fig)


# ==============================================================================
# THÍ NGHIỆM 2: KHẢO SÁT NHIỄU (SNR) VÀ TINH CHỈNH THAM SỐ LAMBDA
# ==============================================================================

def run_exp2_noise_and_stem():
    print("\n=== THÍ NGHIỆM 2: ẢNH HƯỞNG NHIỄU VÀ THAM SỐ LAMBDA (LASSO) ===")
    n, m, k = 256, 100, 12
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    L = power_iteration_lipschitz(A)
    x_true = np.zeros(n)
    supp = rng.choice(n, size=k, replace=False)
    x_true[supp] = rng.standard_normal(k)
    b_clean = A @ x_true

    snrs = [10, 20, 30, 40]
    lambdas = np.logspace(-4, 0, 25)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x_stem_best = None

    for snr in snrs:
        sig_power = np.linalg.norm(b_clean) ** 2 / m
        noise_std = np.sqrt(sig_power / (10 ** (snr / 10)))
        e = noise_std * rng.standard_normal(m)
        b = b_clean + e

        errs = []
        best_err = np.inf
        best_lam = lambdas[0]
        best_k = 0
        for lam in lambdas:
            x_hat, _, _ = fista_lasso(A, b, lam, L, max_iter=600, tol=1e-7)
            err = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
            errs.append(err)
            if err < best_err:
                best_err = err
                best_lam = lam
                best_k = int(np.sum(np.abs(x_hat) > 1e-3))
                if snr == 30:
                    x_stem_best = x_hat.copy()
        print(f"SNR = {snr:2d} dB | lambda* = {best_lam:.4e} | RelErr min = {best_err:.4e} | k_hat = {best_k}")
        ax.loglog(lambdas, errs, marker="o", markersize=3.5, linewidth=1.6, label=f"SNR = {snr} dB")

    ax.set_xlabel(r"Tham số điều hòa $\lambda$", fontsize=10)
    ax.set_ylabel(r"Sai số tương đối $\|\widehat{x}(\lambda) - x^\natural\|_2 / \|x^\natural\|_2$", fontsize=10)
    ax.set_title(r"Đánh đổi sai số khôi phục theo $\lambda$ dưới các mức nhiễu SNR", fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_exp2_noise_lambda.pdf"))
    plt.close(fig)

    # Biểu đồ stem 1D tại SNR = 30 dB
    fig2, ax2 = plt.subplots(figsize=(6.8, 3.4))
    idx = np.arange(n)
    nz_true = np.where(np.abs(x_true) > 1e-6)[0]
    ax2.stem(nz_true, x_true[nz_true], linefmt="b-", markerfmt="bo", basefmt="k-", label=r"Tín hiệu gốc $x^\natural$")
    ax2.plot(idx, x_stem_best, "rx", markersize=5.5, markeredgewidth=1.4, label=r"Khôi phục $\widehat{x}$ (SNR=30dB)")
    ax2.set_xlabel(r"Chỉ số tọa độ $i$", fontsize=10)
    ax2.set_ylabel(r"Biên độ $x_i$", fontsize=10)
    ax2.set_title(r"Khôi phục tín hiệu thưa 1D bằng LASSO (FISTA, $n=256, m=100, k=12$)", fontsize=10.5)
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "fig_exp2_stem_1d.pdf"))
    plt.close(fig2)


# ==============================================================================
# THÍ NGHIỆM 3: SO SÁNH TỐC ĐỘ HỘI TỤ GIỮA ISTA VÀ FISTA
# ==============================================================================

def run_exp3_ista_vs_fista():
    print("\n=== THÍ NGHIỆM 3: SO SÁNH HỘI TỤ ISTA VS FISTA ===")
    n, m, k = 512, 120, 22
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    L = power_iteration_lipschitz(A)
    x_true = np.zeros(n)
    supp = rng.choice(n, size=k, replace=False)
    x_true[supp] = rng.standard_normal(k)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    lam = 0.01 * np.linalg.norm(A.T @ b, np.inf)

    _, F_ref_hist, _ = fista_lasso(A, b, lam, L, max_iter=3000, tol=1e-14)
    F_star = F_ref_hist[-1]

    x_ista, F_ista, t_ista = ista_lasso(A, b, lam, L, max_iter=300, tol=1e-14)
    x_fista, F_fista, t_fista = fista_lasso(A, b, lam, L, max_iter=300, tol=1e-14)

    gap_ista = np.maximum(F_ista - F_star, 1e-15)
    gap_fista = np.maximum(F_fista - F_star, 1e-15)
    print(f"F* = {F_star:.6f}")
    print(f"ISTA  (300 iters): F - F* = {gap_ista[-1]:.3e}, CPU time = {t_ista*1000:.2f} ms")
    print(f"FISTA (300 iters): F - F* = {gap_fista[-1]:.3e}, CPU time = {t_fista*1000:.2f} ms")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.log10(gap_ista), "b--", linewidth=1.8, label=r"ISTA ($\mathcal{O}(1/k)$)")
    ax.plot(np.log10(gap_fista), "r-", linewidth=1.8, label=r"FISTA ($\mathcal{O}(1/k^2)$)")
    ax.set_xlabel(r"Số vòng lặp $k$", fontsize=10)
    ax.set_ylabel(r"$\log_{10}(F(x^k) - F^*)$", fontsize=10)
    ax.set_title(r"Tốc độ hội tụ ISTA vs FISTA trên bài toán LASSO ($n=512, m=120$)", fontsize=10.5)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_exp3_ista_vs_fista.pdf"))
    plt.close(fig)


# ==============================================================================
# THÍ NGHIỆM 4: ẢNH HƯỞNG CỦA MUTUAL COHERENCE
# ==============================================================================

def run_exp4_coherence():
    print("\n=== THÍ NGHIỆM 4: ẢNH HƯỞNG CỦA MUTUAL COHERENCE ===")
    m, n, k = 80, 160, 8
    A = rng.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    G_A = np.abs(A.T @ A)
    np.fill_diagonal(G_A, 0.0)
    mu_A = float(np.max(G_A))

    B = A.copy()
    delta = rng.standard_normal(m)
    delta = 1e-3 * delta / np.linalg.norm(delta)
    B[:, 1] = B[:, 0] + delta
    B[:, 1] /= np.linalg.norm(B[:, 1])
    G_B = np.abs(B.T @ B)
    np.fill_diagonal(G_B, 0.0)
    mu_B = float(np.max(G_B))

    T = 25
    succ_A, succ_B_hit, succ_B_miss = 0, 0, 0
    for _ in range(T):
        # Hỗ trợ chứa chỉ số 0 (trùng cột tương quan cao của B)
        supp_hit = np.concatenate(([0], rng.choice(np.arange(2, n), size=k - 1, replace=False)))
        x1 = np.zeros(n)
        x1[supp_hit] = rng.standard_normal(k)
        if np.linalg.norm(admm_bp(A, A @ x1, max_iter=500) - x1) / np.linalg.norm(x1) < 1e-3:
            succ_A += 1
        if np.linalg.norm(admm_bp(B, B @ x1, max_iter=500) - x1) / np.linalg.norm(x1) < 1e-3:
            succ_B_hit += 1

        # Hỗ trợ tránh cặp {0, 1}
        supp_miss = rng.choice(np.arange(2, n), size=k, replace=False)
        x2 = np.zeros(n)
        x2[supp_miss] = rng.standard_normal(k)
        if np.linalg.norm(admm_bp(B, B @ x2, max_iter=500) - x2) / np.linalg.norm(x2) < 1e-3:
            succ_B_miss += 1

    r_A = 100.0 * succ_A / T
    r_B_hit = 100.0 * succ_B_hit / T
    r_B_miss = 100.0 * succ_B_miss / T
    print(f"mu(A) = {mu_A:.4f} -> Tỷ lệ thành công: {r_A:.1f}%")
    print(f"mu(B) = {mu_B:.6f} -> Khi hỗ trợ chứa cột 1: {r_B_hit:.1f}% | Khi tránh {{1,2}}: {r_B_miss:.1f}%")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = [f"Ma trận A\n($\\mu={mu_A:.3f}$)",
              f"Ma trận B (chứa cột 1)\n($\\mu\\approx {mu_B:.5f}$)",
              f"Ma trận B (tránh cột 1,2)\n($\\mu\\approx {mu_B:.5f}$)"]
    vals = [r_A, r_B_hit, r_B_miss]
    colors = ["#2b5c8f", "#c0392b", "#27ae60"]
    bars = ax.bar(labels, vals, color=colors, width=0.52)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Tỷ lệ khôi phục thành công (%)", fontsize=10)
    ax.set_title(r"Ảnh hưởng của Mutual Coherence đến tỷ lệ khôi phục ($m=80, n=160, k=8$)", fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_exp4_coherence.pdf"))
    plt.close(fig)


# ==============================================================================
# THÍ NGHIỆM 5: NÉN VÀ TÁI TẠO ẢNH 2D VỚI BIẾN ĐỔI WAVELET DB4
# ==============================================================================

def compute_psnr(img_ref: np.ndarray, img_rec: np.ndarray) -> float:
    mse = float(np.mean((img_ref - img_rec) ** 2))
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - mx) * (y - my)))
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))


def make_synthetic_phantom(size: int = 64) -> np.ndarray:
    """Tạo ảnh phantom hình học 2D chuẩn hóa [0, 1] (thưa trong miền Wavelet)."""
    yy, xx = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    img = np.zeros((size, size))
    img[(xx / 0.75) ** 2 + (yy / 0.90) ** 2 <= 1.0] = 0.85
    img[(xx / 0.65) ** 2 + (yy / 0.80) ** 2 <= 1.0] = 0.25
    img[((xx - 0.22) / 0.16) ** 2 + ((yy + 0.1) / 0.28) ** 2 <= 1.0] = 0.0
    img[((xx + 0.22) / 0.16) ** 2 + ((yy + 0.1) / 0.28) ** 2 <= 1.0] = 0.0
    img[(np.abs(xx) < 0.25) & (np.abs(yy - 0.35) < 0.10)] = 0.95
    return img


def run_exp5_image_wavelet():
    print("\n=== THÍ NGHIỆM 5: NÉN VÀ TÁI TẠO ẢNH 2D WAVELET DB4 ===")
    N_side = 64
    n = N_side * N_side
    img = make_synthetic_phantom(N_side)

    # Xây dựng biến đổi Wavelet 2D (db4, level 3)
    coeffs0 = pywt.wavedec2(img, "db4", mode="periodization", level=3)
    arr0, slices = pywt.coeffs_to_array(coeffs0)
    wav_shape = arr0.shape
    n_wav = arr0.size

    def wav_rec(theta_vec):
        c = pywt.array_to_coeffs(theta_vec.reshape(wav_shape), slices, output_format="wavedec2")
        return pywt.waverec2(c, "db4", mode="periodization")[:N_side, :N_side]

    def wav_dec(im):
        c = pywt.wavedec2(im, "db4", mode="periodization", level=3)
        v, _ = pywt.coeffs_to_array(c)
        return v.ravel()

    results = []
    for ratio in [0.3, 0.5]:
        m = int(ratio * n)
        A = rng.standard_normal((m, n)) / np.sqrt(m)
        b = A @ img.ravel()

        # Giải LASSO trong miền Wavelet bằng FISTA với toán tử tuyến tính A * Psi
        L = power_iteration_lipschitz(A, n_iters=20)
        lam = 0.004
        theta = np.zeros(n_wav)
        y_var = theta.copy()
        t = 1.0
        step = 1.0 / L
        tau = lam * step
        for _ in range(250):
            x_cur = wav_rec(y_var).ravel()
            res = A @ x_cur - b
            grad_x = (A.T @ res).reshape(N_side, N_side)
            grad_theta = wav_dec(grad_x)
            theta_next = soft_threshold(y_var - step * grad_theta, tau)
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y_var = theta_next + ((t - 1.0) / t_next) * (theta_next - theta)
            theta = theta_next
            t = t_next

        img_rec = np.clip(wav_rec(theta), 0.0, 1.0)
        psnr = compute_psnr(img, img_rec)
        ssim = compute_ssim(img, img_rec)
        print(f"Tỷ lệ đo m/n = {ratio:.1f} (m = {m}/{n}) -> PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
        results.append((ratio, img_rec, psnr, ssim))

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(r"Ảnh gốc ($64 \times 64$)", fontsize=10)
    axes[0].axis("off")

    for idx_ax, (ratio, img_rec, psnr, ssim) in enumerate(results, start=1):
        axes[idx_ax].imshow(img_rec, cmap="gray", vmin=0, vmax=1)
        axes[idx_ax].set_title(f"$m = {ratio:.1f}n$\nPSNR = {psnr:.2f} dB, SSIM = {ssim:.3f}", fontsize=9.5)
        axes[idx_ax].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_exp5_image_cs.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    t_start = time.perf_counter()
    run_exp1_phase_transition()
    run_exp2_noise_and_stem()
    run_exp3_ista_vs_fista()
    run_exp4_coherence()
    run_exp5_image_wavelet()
    print(f"\nHoàn tất toàn bộ 5 thí nghiệm sau {time.perf_counter() - t_start:.2f} giây.")
