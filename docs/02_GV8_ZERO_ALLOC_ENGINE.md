# Đặc tả Động cơ Suy luận Rust AVX2 `GV8MOD01`: Nén Full-RAM Q8, Không Cấp phát Heap & Độ sâu Tư duy An toàn Trạng thái (`#87`)

> **Thuộc chuỗi đặc tả**: `Genesis-v8 BICA v2` (`#86`–`#91`) | **Sửa trực tiếp các phát hiện**: `C1, C2, C3, C4, C5, C10`  
> **Ràng buộc kỹ thuật**: 100% Rust thuần (`rustc`, 0 crate ngoài), 100% CPU AVX2+FMA offline, mọi tệp `.rs`/`.py` mới `<= 400` LOC, `0 B` trên ổ `C:\`.

---

## 1. Động lực & Bằng chứng Đo đạc Thực tế (`i5-10210U` + `v7_ultra_p1_preview.bin`)

Đo trực tiếp trên checkpoint thật [`v7_ultra_p1_preview.bin`](../models/genesis_chat_3m/v6_lm/v7_ultra_p1_preview.bin) (`982.72 MB`, `GV6MOD03`, `445.96M` tham số) qua `genesis_v6.dll`:

| Chỉ số Hiện trạng (`GV6MOD03`) | Giá trị Thực đo / Kiểm toán Mã nguồn | Nguyên nhân Gốc trong Mã nguồn |
| :--- | :---: | :--- |
| **Thời gian nạp tệp (Cold Load)** | **`2,370.6 ms`** | `genesis_pro_chat.py:75-89` gọi `subprocess.run` nạp lại toàn bộ `982.7 MB` mỗi lượt |
| **RAM tiến trình (RSS / Commit)** | **`977.0 MiB` / `979.6 MiB`** | `BitLinear` (`model.rs:44-49`) giữ kép `w:f32` (`161.61 MB`) + `q:i8` (`40.40 MB`); `PEER` (`peer_pk.rs:94-98`) giữ `i16` (`805.31 MB`) + `vu,vv` (`4.19 MB`) |
| **Tốc độ sinh ấm (`genesis_v6.dll`)** | **`R=1: 505.7 tok/s` \| `R=2: 343.5 tok/s` \| `R=4: 241.4 tok/s`** | Active thật `10.31M` (`R=1`) / `14.10M` (`R=2`), active bytes **`19.83 MB/tok`** |
| **Cấp phát Heap trong vòng lặp token** | **`12` `Vec` / token × bước `R`** | `7` ở `attn_rope.rs:108-133`, `4` ở `peer_pk.rs:206-209`, `1` ở `generate.rs:100` (`exec_plan`) |
| **Phân phối định tuyến PEER thật** | **Zipf `PEER_0: s=0.618`, `PEER_1: s=0.654`** | Không nhọn như `s=1.05`: cache 15% RAM chỉ trúng **`49.6%`**! |
| **NVMe `E:\` Uncached 4KB Read (`QD=1`)** | **`0.1781 ms` (`5,614 IOPS`)** | `u_exp`, `v_exp` tách 2 mảng → 2 seek/miss (`0.356 ms`/miss) ⇒ **CẤM phân trang SSD cho `ultra_445m`!** |
| **Đổi `R` giữa chừng & ngoại suy `R=4`** | **`R=1: ppl=72.6 → R=2: 74.2 → R=4: 117.3`** | `ffi.rs:73-79` xóa sạch `h.state`; `model.rs:103-111` tách KV/MinGRU riêng theo từng độ sâu `step` |

---

## 2. Thiết kế Định dạng Nhị phân `GV8MOD01`

### 2.1. Chế độ Khuyến nghị Ngay cho `ultra_445m` (`Genesis-v8A`: Full-RAM Q8, `0` SSD Miss)
- **Magic Header**: `b"GV8MOD01"` + struct cấu hình `ModelCfg` + cờ chế độ lượng tử hóa (`bitnet_fmt`, `peer_fmt`, `dense_fmt`).
- **Xóa sạch trạng thái huấn luyện khi suy luận**: Không lưu/nạp `BitLinear.w: f32` (`-161.61 MB`) và không nạp `PEER vu, vv` (`-4.19 MB`) vào tiến trình suy luận (chuyển riêng sang `sleep_peer_delta.bin` trong `#91`).
- **Nén `BitLinear` sang 2-bit đóng gói (`I2_S` / `TL2` LUT)**:
  - Trọng số tam phân $q \in \{-1, 0, +1\}$ ánh xạ bit-exact thành `{00: 0, 01: +1, 10: -1}` (4 trọng số/byte, giảm `40.40 MB → 10.10 MB`, `rel_err == 0.0`).
  - *Vi-chuẩn bắt buộc trước khi khóa nhân*: So sánh trực tiếp chu kỳ CPU trên `i5-10210U` giữa `I2_S`/`TL2` LUT và `dot_ternary_avx2_inner` (`i8`) ở kích thước `384×384` và `512×384` (độ thưa số 0 đo thực `37.70%`). Nếu unpack 2-bit chậm hơn `i8` tại lớp nhỏ, giữ `2-bit` trên đĩa và unpack `i8` cho ma trận nhỏ (`<= 40 MB`) hoặc dùng bảng tra AVX2 `_mm256_shuffle_epi8`.
- **Nén `PEER` sang `Q8 per-row` Full-RAM (`404.70 MiB` tổng cả mô hình!)**:
  - Mỗi hàng chuyên gia $u_e, v_e \in \mathbb{R}^{384}$ lưu `i8[384]` + hệ số tỷ lệ `f16`/`f32` theo từng hàng (`scale_u[e]`, `scale_v[e]`), giảm bộ nhớ PEER từ **`805.31 MB → 402.65 MB`** (`~384 MiB`), nằm gọn hoàn toàn trong RAM (`404.70 MiB` tổng trọng số `ultra_445m`!) với **`0` SSD miss** và giữ trọn tốc độ **`350–520+ tok/s`**!
- **Tùy chọn `Genesis-v8B` (Giảm Active Bytes từ `14.27 MB/tok` → `~5.7 MB/tok`)**:
  - Lượng tử hóa đối xứng theo hàng `int8` (hoặc `f16`) cho `lm_head` (`6.29 MB → 1.57 MB`), `Attn W_qkv/W_o` (`3.15 MB → 0.80 MB`), và `PEER wq/keys` (`2.75 MB → 0.70 MB`) kèm bước kiểm định `ΔNLL <= 0.015`.

### 2.2. Chế độ Mở rộng Tương lai `mega_1.6B` (`Genesis-v8C`: Interleaved SSD + Async IOCP)
Chỉ kích hoạt khi mô hình vượt `1.0B` tham số (`PEER > 1.2 GB`):
1. **Đóng gói xen kẽ `[u_e, v_e]` liên tiếp (`768 B/chuyên gia`)**: Giảm từ 2 lần tìm đĩa (`0.356 ms`) xuống đúng **1 lần đọc** (`0.178 ms`) mỗi cache miss.
2. **Windows Overlapped Async I/O (`IOCP` / `ReadFileEx` với `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, `QD=16..32`)**: Phát đồng thời toàn bộ $K_{\text{miss}}$ yêu cầu đọc trong 1 token thay vì chờ tuần tự `QD=1`, kết hợp bộ đệm LRU nóng `15%` RAM (~`230 MiB`).

---

## 3. Khử 100% Cấp phát Heap (`Zero-Alloc step_token`) & Persistent Daemon

### 3.1. Kiến trúc `StepScratch` Không Cấp phát (`0` `Vec::new` / Token)
Chuyển toàn bộ **12 `Vec` đang cấp phát mỗi token** vào một struct cấp phát sẵn một lần trong `InferenceState`:
- `attn_rope.rs:108-133` (**7 buffer**): `attn_xn[d]`, `attn_q[d]`, `attn_k[d]`, `attn_v[d]`, `attn_out[d]`, `attn_scores[W]`, và bảng tra **`inv_freq[hd/2]` tính sẵn 1 lần khi tạo `InferenceState`** (xóa hoàn toàn `powf()` lặp lại mỗi token!).
- `peer_pk.rs:206-209` (**4 buffer**): `peer_xn[d]`, `peer_q[n_heads*sub_d]`, `peer_sa[sub_k]`, `peer_sb[sub_k]`.
- `generate.rs:100` (**1 buffer**): `exec_plan: [usize; MAX_PLAN_LEN]` cố định theo `think_steps` (tính sẵn khi gọi `set_think_steps`).

### 3.2. Persistent `V6RustLMHandle` Daemon & Multi-Core Tree-of-Thought
- Thay toàn bộ `subprocess.run` trong `mel/v6_lm/` bằng handle DLL giữ sống trên RAM (`mel/v6_lm/rust_v6_bridge.py` + `genesis_v6_rust/src/reason_tree.rs` — đã biên dịch & xác chứng giảm 8 nhánh song song từ `~32.6s` xuống `0.83–1.15s`, tăng tốc **~39.5×**!).
- Sửa lỗi gọi kép `vault.retrieve()` tại `mel/v6_lm/genesis_pro_chat.py:69-71` (**đã vá**, tiết kiệm `28.89 ms`/truy vấn).

---

## 4. Độ sâu Suy nghĩ An toàn Trạng thái (`R` Turn-Level cho `v7` → `Huginn/HRM` cho `v8`)

### 4.1. Pha I — An toàn cho `v7_ultra` (0 Retrain)
- Cấp phát sẵn `InferenceState` với `R_max = 2` (tất cả slot `step=0` và `step=1` tồn tại liên tục).
- Khi gọi `v6_set_think_steps(handle, r)` với `r ∈ {1, 2}`: **KHÔNG xóa `h.state`**, chỉ cập nhật `st.think_steps = r.clamp(1, st.max_think_steps)` và cập nhật `exec_plan` tại chỗ.
- Chuyển đổi `R=1 ↔ R=2` chỉ thực hiện ở **ranh giới lượt (turn-level)** hoặc khi có `<CTX>` mới (tránh hụt lịch sử KV-cache ở `step=1`).

### 4.2. Pha II — Độ sâu Thích ứng Từng Token `R=1..8` cho `Genesis-v8` Pretraining
- **Shared-KV Recurrent Depth (*Huginn*, Geiping et al. 2025)**: Mọi bước lặp độ sâu `r = 1..R` đọc/ghi **chung 1 KV-cache** và chuẩn hóa dòng dư có cổng; khi huấn luyện trên H100 lấy mẫu độ sâu ngẫu nhiên $R \sim \text{LogNormal}(\mu, \sigma)$ để không bao giờ sập perplexity khi ngoại suy lên `R=4..8`.
- **Phân tách Hai Thang Thời gian (*HRM*, Sapient 2025)**: Tách vòng **Lập kế hoạch Chậm (High-Level Cortex, cập nhật mỗi $K$ bước)** và vòng **Thực thi Nhanh (Low-Level Cortex)** nhằm giảm chi phí suy luận sâu.

---

## 5. Cổng Kiểm định & Tiêu chí Bác bỏ (Acceptance / Falsification Gates)

1. **BitNet 2-bit Parity**: `max_abs_diff == 0.0`, `rel_err == 0.0` so với `i8` trên 1,000 token mẫu.
2. **PEER `Q8 per-row` Functional Equivalence (đã hiệu đính `C4`)**: `top1_match >= 98.5%`, `ΔNLL <= 0.01`, `logit_rel_err <= 0.02` trên held-out `val` (giữ tùy chọn `Q14` lossless nếu cần `< 1e-3`).
3. **Ngân sách RAM `ultra_445m`**: Tổng RSS của handle `GV8MOD01` Full-RAM Q8 **`<= 430 MiB`** (`0` SSD miss).
4. **Zero-Alloc**: Đếm bộ cấp phát tùy biến (`GlobalAlloc` test hook) xác nhận **`0` lượt `alloc/dealloc`** trong suốt `step_token`.
5. **Tốc độ Laptop `i5-10210U`**: `>= 500 tok/s` (`R=1`) và `>= 350 tok/s` (`R=2`) ở chế độ đơn luồng ấm; `>= 350 eff-tok/s` khi chạy 4 nhánh song song `reason_tree.rs`.
6. **Bất biến Không Hồi quy**: `v6_set_think_steps(1 -> 2 -> 1)` giữ nguyên 100% lịch sử `step=0` và không làm sai lệch đầu ra so với phiên liên tục.
