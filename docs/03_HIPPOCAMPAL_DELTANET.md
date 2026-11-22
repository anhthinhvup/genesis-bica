# Đặc tả Bộ nhớ Nhanh Hai Tầng: Kanerva/Hopfield 1-Shot (0 Retrain) & Gated DeltaNet / Titans HOPE / Engram Bản địa (`#88`)

> **Thuộc chuỗi đặc tả**: `Genesis-v8 BICA v2` (`#86`–`#91`) | **Sửa trực tiếp phát hiện**: `C6` (sụp đổ Delta-Rule trên trạng thái ẩn dị hướng thật của `v7_ultra` & phân tách học thuật `Gated DeltaNet` ≠ `Titans` ≠ `HOPE`)  
> **Ràng buộc kỹ thuật**: 100% CPU offline, mọi tệp mã nguồn mới `<= 400` LOC, `0 B` ghi ổ `C:\`, `0` vi phạm Gate-1 AST.

---

## 1. Bằng chứng Phản chứng Quan trọng Trên Trạng thái Ẩn Thật (`C6`)

Kiểm toán thực nghiệm trên [`v7_ultra_p1_preview.bin`](../models/genesis_chat_3m/v6_lm/v7_ultra_p1_preview.bin) chứng minh một cạm bẫy kinh điển của thí nghiệm đồ chơi:
- **Vector Gaussian trực giao (giả lập)**: Delta-Rule đạt `cos = 0.9945`, 1-shot overwrite `1.0000`, giữ lại `0.9915`.
- **Trạng thái ẩn THẬT của `v7_ultra_p1_preview.bin`** có tính **dị hướng (anisotropy) mạnh**:
  - `Post-Attn`: `mean_cos = 0.1785` (`p99 = 0.477`).
  - `Layer 5`: `mean_cos = 0.2580` (`p99 = 0.629`).
- **Kết quả đo thực nghiệm đối chứng (Verify 2 trên `i5-10210U`)**:

| Cơ chế Bộ nhớ Nhanh (trên trạng thái thật `v7_ultra`, `N=64` / `N=128`) | Độ chính xác Truy hồi (`cos`) | Độ giữ sau Ghi đè 1-Shot | Độ trễ / Truy vấn | Khả thi Post-Hoc (0 Retrain)? |
| :--- | :---: | :---: | :---: | :---: |
| **Delta-Rule 1-pass thô** | `0.7498` (`N=64`) / `0.6951` (`N=128`) | `0.7433` (Sụp!) | `~790 µs` (NumPy) | **KHÔNG** (nhiễu dị hướng + chưa có $W_k, W_v$) |
| **Titans MLP 2 tầng ngẫu nhiên** | `0.6049` (Sụp!) | Kém | `412 µs` | **KHÔNG** (bắt buộc meta-train đầu-cuối!) |
| **Gated DeltaNet chuẩn hóa tâm tĩnh** | **`0.9586`** | Tốt | `< 50 µs` (Rust) | Chỉ nên dùng khi **pretrain `Genesis-v8`** |
| **Centered Modern Hopfield / Kanerva SDM** | **`1.0000`** | **`1.0000`** (Khe rời + ATMS!) | **`5.5 µs`** | **CÓ — TỐI ƯU CHO `v7_ultra` NGAY LẬP TỨC!** |

**Kết luận kiến trúc bất di bất dịch**: **TUYỆT ĐỐI KHÔNG** tiêm ma trận `M_t` chưa được huấn luyện vào dòng dư đóng băng của `v7_ultra`. Phải tách rạch ròi **Tầng 1 (0 Retrain cho `v7_ultra`)** và **Tầng 2 (Pretrain Bản địa cho `Genesis-v8`)**.

---

## 2. Tầng 1 — Bộ nhớ Khe Hiện tường Tức thời cho `v7_ultra` (0 Retrain, `5.5 µs`)

Hợp nhất 3 thành phần đã có nền tảng vững trong `Project Genesis` thành 1 mô-đun thống nhất `mel/v6_lm/fast_episodic_memory.py` (`<= 350` LOC) chạy trong không gian làm việc toàn cục (`#90`):

1. **Khử Dị hướng (Whitening / Running-Mean Centering)**:
   Với mọi khóa sự kiện $k_t \in \mathbb{R}^d$ (trích từ token/biểu diễn trạng thái ẩn `Post-Attn` hoặc vector ngữ nghĩa nhẹ), tính khóa đã căn tâm và chuẩn hóa đơn vị:
   $$\hat{k}_t = \frac{k_t - \bar{\mu}}{\|k_t - \bar{\mu}\|_2 + \epsilon}$$
   triệt tiêu thành phần nón dị hướng `mean_cos = 0.1785 → ~0.0`.
2. **Bộ nhớ Khe Hiện tường (`Kanerva SDM` + `Modern Hopfield` nhiệt độ cao $\beta$)**:
   - Lưu danh sách khe có thời hạn & trọng số độ bất ngờ $\{( \hat{k}_i, v_i^{\text{text}}, s_i, t_i, \text{status}_i )\}_{i=1}^{M}$ ($M \le 4{,}096$ khe nóng trong RAM $\approx 6\text{ MB}$, lưu bền vững xuống SQLite/binary trên `E:\learn\project_genesis\.cache_e\episodic_slots.db`).
   - Truy hồi theo mật độ Hopfield:
     $$p_i(q) = \frac{\exp(\beta \, \hat{q}^\top \hat{k}_i)}{\sum_{j \in \text{ACTIVE}} \exp(\beta \, \hat{q}^\top \hat{k}_j)}$$
     Đạt **`cos = 1.0000` trong `5.5 µs/truy vấn`** trên CPU!
3. **Sửa Niềm tin Phi Đơn điệu 1-Shot (`ATMS Non-Monotonic Revision`)**:
   - Khi người dùng đính chính sự kiện (ví dụ *"Thực ra tôi đã chuyển sang Đà Nẵng"*), đối sánh khe cũ tương đồng cao + cùng thực thể chủ đề qua `v5_episodic_belief_revision.py` → đánh dấu khe cũ `SUPERSEDED` (giữ lịch sử thời gian trong sổ cái, không trả vào ngữ cảnh hiện tại) và chèn khe mới `ACTIVE` với độ bất ngờ cao.
   - Kết quả truy hồi được bơm sạch vào khối `<CTX>` (Memory-as-Context) cho Bán cầu Phải `v7_ultra` và cung cấp trực tiếp cho Bán cầu Trái (`SelfModelGraph` / `Sheaf`).

---

## 3. Tầng 2 — Bộ nhớ Tham số Bản địa Cho `Genesis-v8` Pretraining trên H100

Khi khởi chạy thế hệ tiền huấn luyện mới `Genesis-v8`, nâng cấp trực tiếp kiến trúc lõi tuần hoàn theo 3 công trình nền tảng 2025–2026 (phân biệt rõ từng cơ chế, không đánh đồng):

### 3.1. `Multi-Head Gated DeltaNet` (*Yang et al., ICLR 2025*)
Thay thế các lớp `MinGRU` tĩnh bằng ma trận trạng thái có cổng quên $\alpha_t \in (0, 1)$, cổng học $\beta_t \in (0, 1)$ và **chuẩn hóa L2 khóa $k_t$** (chặn đứng hiện tượng sụp đổ dị hướng):
$$S_t = \alpha_t S_{t-1}\left(I - \beta_t \hat{k}_t \hat{k}_t^\top\right) + \beta_t v_t \hat{k}_t^\top, \qquad \|\hat{k}_t\|_2 = 1$$
- *Chi phí RAM trên laptop*: Với `H=6` đầu, `d_k = d_v = 64` mỗi lớp: $6 \times 64 \times 64 \times 4\text{ B} = \mathbf{96\text{ KB/lớp}}$ (6 lớp = **`576 KB` tổng cộng** — nhẹ hơn 6 lần ma trận toàn cục `384×384` mà vẫn song song hóa chunk-wise cực nhanh trên H100!).

### 3.2. `Titans MAC` & `Nested Learning / HOPE` (*Behrouz et al., 2024 & NeurIPS 2025, arXiv:2512.24695*)
- **Titans Neural Long-Term Memory**: Dùng bộ nhớ MLP sâu 2 tầng cập nhật lúc suy luận theo **độ bất ngờ có động lượng (Momentum Surprise)**:
  $$S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; k_t, v_t), \qquad \mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t$$
  kết hợp theo mô hình **Memory-as-Context (`MAC`)** để không phá vỡ dòng dư.
- **HOPE Continuum Memory System**: Xếp chồng đa thang thời gian (nhanh từng token → trung bình từng phiên → chậm trong `PEER` lúc ngủ `#91`) thành một chuỗi tối ưu hóa lồng nhau thống nhất.

### 3.3. Bộ nhớ Có điều kiện `Engram` (*DeepSeek, 1/2026*)
- Gắn bảng băm $N$-gram đa đầu truy cập $O(1)$ tất định tại tầng `0` và tầng `1` (chiếu vector gọn $d_{\text{eng}} \to d$ qua cổng nhân):
  - Bảng băm nằm trên DRAM hoặc `mmap` trên ổ `E:\` (`~100–250 MB`), truy xuất $O(1)$ không cần tính tích ma trận.
  - Phối hợp trực tiếp với `FM-InfiniGram (#63)` sẵn có trong `Genesis`, giải phóng các tầng đầu khỏi nhiệm vụ ghi nhớ cụm từ cố định để dồn năng lực của lõi `d=384` cho suy luận trừu tượng.

---

## 4. Cổng Kiểm định & Tiêu chí Bác bỏ (Acceptance / Falsification Gates)

1. **Tầng 1 (0 Retrain trên `v7_ultra`)**:
   - Đạt **`>= 98%` độ chính xác Sửa Niềm tin 1-Shot** trên bộ kiểm thử 50 ca đổi ý liên tiếp (gồm cả đối tượng trùng tên và mâu thuẫn thời gian).
   - Độ trễ truy hồi `< 0.5 ms/truy vấn` trên `i5-10210U` (`M = 4,096` khe); RAM `< 12 MB`.
   - **Zero-Tradeoff**: `0%` hồi quy trên `Adversarial-312` (`>= 301/312, FakeVerify=0`) và `Parity-500` (`>= 489/518`).
2. **Tầng 2 (`Genesis-v8` Native Pretraining)**:
   - Vượt `MinGRU` ít nhất `+15 pp` trên tác vụ **Needle-in-a-Haystack / Associative Recall** ở độ dài ngữ cảnh `2,048–4,096` tokens trong cùng ngân sách FLOPs huấn luyện.
   - Bảo toàn trạng thái liên phiên sau khi serialize/deserialize xuống `E:\` với sai số bit `0.0`.
