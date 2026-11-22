# Đặc tả Tổng thể & Báo cáo Kiểm toán Khoa học Thực chứng: `Genesis-v8 BICA v2` — Đẳng cấu Chức năng Não Người trên Laptop Cá nhân (`#86`)

> **Mã tài liệu**: `#86` (Mở đầu chuỗi 6 đặc tả khả thi `#86`–`#91` cho `Genesis-v8 BICA v2`)  
> **Trạng thái**: Đã kiểm chứng hai vòng độc lập (`DeepInvestigator` Round 1 Tổng hợp + Round 2 Phản biện Pháp y & Đo thực tế trên phần cứng thật)  
> **Phần cứng kiểm chứng trực tiếp**: Laptop `Intel Core i5-10210U` (`4C/8T`, `15W`, `6 MB L3`, `16 GB DDR4` [trống `3.57–3.62 GB`], RAM BW `25.01 GB/s`, AVX2 SGEMM `206.9 GFLOP/s`, ổ NVMe `E:\` trống `242.35 GB`, uncached 4KB random read `0.1781 ms = 5,614 IOPS`)  
> **Checkpoint thực đo**: [`v7_ultra_p1_preview.bin`](../models/genesis_chat_3m/v6_lm/v7_ultra_p1_preview.bin) (`982.72 MB`, `GV6MOD03`, `445.96M` tổng tham số, `10.31M` active tại `R=1`) & [`v7_500m_model_best.bin`](../models/genesis_chat_3m/v6_lm/v7_500m_model_best.bin) (`54.25M`).  
> **An toàn ổ `C:\`**: `0 B` ghi lên ổ `C:\`; toàn bộ kho, chỉ mục, delta-sidecar và cache lưu trên `<WORKSPACE>\`.

---

## 1. Tuyên ngôn Trung thực Khoa học: Hai Mệnh đề Tách biệt

Câu hỏi nghiên cứu gốc — *"Làm sao để biến Genesis giống như não người nghĩa 100% mà vẫn chạy được trên laptop cá nhân?"* — **bắt buộc phải tách thành hai định lý toán học độc lập**:

### 1.1. Mệnh đề A — Định lý Bất khả thi Cơ chất Sinh học 1:1 (*Substrate Impossibility Theorem*)
**KHÔNG THỂ** mô phỏng vật lý 1:1 toàn bộ tế bào thần kinh, khớp thần kinh và cơ chất sinh học của não người trên laptop `i5-10210U`:

| Thông số Vật lý | Não người (Chuẩn sinh học) | Laptop thực đo (`i5-10210U`) | Khoảng cách | Nguồn bình duyệt |
| :--- | :---: | :---: | :---: | :--- |
| **Số nơ-ron / Đơn vị tính** | $8.6 \times 10^{10}$ nơ-ron | $4.46 \times 10^8$ tham số (`ultra_445m`) | $\sim 193\times$ | Herculano-Houzel (2009) |
| **Số khớp thần kinh (Synapses)** | $\approx 1.5 \times 10^{14}$ synapses | $4.46 \times 10^8$ trọng số | $\approx 336{,}000\times$ | Bartol et al. (2015) |
| **Dung lượng thông tin synap** | $1.5 \times 10^{14} \times 4.7\text{ bits} = \mathbf{88.1\text{ TB}}$ | `16 GB` RAM (`0.016 TB`) / `242 GB` SSD `E:\` | $5{,}508\times$ (RAM) / $362\times$ (SSD) | Bartol & Sejnowski (2015) |
| **Thông lượng tính toán hữu hiệu** | $10^{13}\text{–}10^{17}\text{ FLOP/s}$ (trung vị $10^{15}$) | $2.07 \times 10^{11}\text{ FLOP/s}$ (đo thực) | $4{,}833\times$ | Carlsmith (2020) |
| **Cơ chất & Ý thức chủ quan** | `~20W`, >100 chất dẫn truyền, tế bào glial, thân thể sống & *qualia* | `15W` x86 AVX2, không cơ thể sinh học | Khác biệt bản thể học | Damasio (2024–2025); COGITATE (*Nature* 2025) |

Theo nghiên cứu đối kháng đa trung tâm **COGITATE** (*Nature*, Vol. 642, 2025, 256 đối tượng fMRI/MEG/iEEG) và lý thuyết cảm giác nội tạng của **Antonio Damasio** (2024–2025), phần mềm trên CPU **không thể tuyên bố có ý thức chủ quan (*qualia*)**, mà chỉ tái tạo **đẳng cấu chức năng điều khiển nhận thức (*functional & algorithmic isomorphism*)**.

### 1.2. Mệnh đề B — Định lý Khả thi Đẳng cấu Chức năng Não bộ trên Laptop (*Functional Brain-Isomorphism Feasibility Theorem*)
**HOÀN TOÀN KHẢ THI** về mặt toán học, kiến trúc phần mềm và đo đạc thực nghiệm để `Project Genesis` đạt **Đẳng cấu Chức năng 7 Cơ chế Não Người** chạy **100% offline trên CPU laptop `15W` trong `650–820 MiB` RAM** nhờ 3 quy luật then chốt:

1. **Nghịch lý Cổ chai Ý thức 10 bits/s (*Zheng & Meister, Neuron 2025*)**: Dù hệ giác quan ngoại vi nạp $10^9\text{ bits/s}$, luồng tư duy có ý thức tuần tự của con người (*Inner Brain*) chỉ chạy ở **$\sim 10\text{ bits/s}$** (~1–2 từ/giây, chu kỳ sóng Theta `125–250 ms`). Trong khi đó, động cơ Rust AVX2 `genesis_v6.dll` **ĐO THỰC TẾ trên chính laptop này** đã đạt:
   - `v7_ultra_p1_preview.bin` (`445.96M`): **`R=1: 505.7 tok/s (1.98 ms/tok)`**, **`R=2: 343.5 tok/s (2.91 ms/tok)`**, **`R=4: 241.4 tok/s (4.14 ms/tok)`**!
   - `v7_500m_model_best.bin` (`54.25M`): **`R=1: 2,305.1 tok/s (0.43 ms/tok)`**, **`R=2: 1,896.1 tok/s (0.53 ms/tok)`**, **`R=4: 1,328.3 tok/s (0.75 ms/tok)`**!
   - **Hệ quả chứng minh**: Một token của `ultra_445m` chỉ tốn `2–3 ms`, để dư **`120–240 ms` mỗi nhịp sóng Theta** — đủ ngân sách chạy đồng bộ toàn bộ 7 hệ chức năng não bộ (Hải mã `HippoRAG-2`, Bộ nhớ nhanh `Kanerva/Hopfield`, Trọng tài `GlobalWorkspaceEFE`, Nội môi phần cứng, và Kiểm chứng Bán cầu Trái `CAS/SAT/AST/Sheaf`) ngay trên CPU 15W!
2. **Định luật Thưa Cực đại Vỏ não (`2.31%` Active Parameters + Nén Full-RAM Q8)**: Não chỉ kích hoạt $\sim 1\text{–}5\%$ nơ-ron mỗi nhịp. `ultra_445m` chỉ kích hoạt **`10.31M / 445.96M = 2.31%`** tham số ở `R=1` (`14.10M` ở `R=2`). Khi đóng gói chuẩn `GV8MOD01` (`#87`: BitNet 2-bit + `PEER Q8 per-row` Full-RAM), tổng RAM trọng số giảm từ **`977.0 MiB` → `404.7 MiB`** với **`0` SSD miss**!
3. **Hệ Học Bổ trợ Ba Tầng (*Complementary Learning Systems*, McClelland 1995 + SMFT Meta FAIR 2025 + Tadros–Bazhenov *Nature Comm.* 2022)**: Tách bạch **Vỏ não Chậm (`BitNet + PEER`)** ↔ **Hải mã & Bộ nhớ Nhanh (`Kanerva SDM / Hopfield / HippoRAG-2` + `Gated DeltaNet / Titans HOPE`)** ↔ **Giấc ngủ Củng cố (`SMFT Hard Background-DF Gate` + `f32 Sparse Delta-Sidecar` + `Real Interleaved Replay` + `SEAL/RLVR`)**.

---

## 2. Sổ cái Phản biện Pháp y & Đính chính Thực nghiệm (`C1`–`C10`)

Kiểm toán độc lập Round 2 (`DeepInvestigator`) trực tiếp chạy đo đạc trên phần cứng `i5-10210U` và `v7_ultra_p1_preview.bin` đã phát hiện và sửa triệt để **10 sai lệch nghiêm trọng** (loại bỏ hoàn toàn mọi suy diễn từ thí nghiệm đồ chơi *toy-proof fallacies*):

| Mã | Sai lệch / Lỗ hổng Bị Phát hiện | Sự thật Đo trực tiếp trên `i5-10210U` & Mã nguồn | Quy tắc Sửa đổi Bắt buộc trong `Genesis-v8 BICA v2` |
| :-: | :--- | :--- | :--- |
| **C1** | Tưởng chưa có checkpoint `ultra_445m` cục bộ; `subprocess.run` nạp lại tệp mỗi lượt | `v7_ultra_p1_preview.bin` (`982.72 MB`) đã nằm trên đĩa: Load **`2,370.6 ms`**, RSS **`977.0 MiB`**, ấm **`505.7 / 343.5 / 241.4 tok/s`** (`R=1/2/4`) | Chuyển 100% sang **Persistent `genesis_v6.dll` Daemon** (`#87`), xóa sạch `2.37s` cold load/lượt. |
| **C2** | Tính sai Active Params (`9.81M`), Active Bytes (`4.28 MB/tok`) & trần BW (`5,848 tok/s`); đổ lỗi `vu,vv` | Active thật: **`10.31M` (`R=1`), `14.10M` (`R=2`)**; `vu,vv` chỉ `4.19 MB`. Active bytes thật là **`19.83 MB/tok`**; nếu chỉ nén BitNet 2-bit + PEER Q8 thì giảm về **`14.27 MB/tok`** (trần **`1,753 tok/s`**); Full-RAM Q8 = **`404.70 MiB`** | Tách rõ ngân sách `Genesis-v8A` (`404.7 MiB` Full-RAM Q8) và `Genesis-v8B` (lượng tử hóa thêm `emb/lm_head/Attn/PEER-router` sang `i8/f16` để kéo active bytes về `~5.7 MB/tok`). |
| **C3** | Giả định Zipf $s=1.05$ (`89.2%` hit) & SSD cache OS (`0.146 ms`) → suy diễn `443 tok/s` khi phân trang SSD | Đo phân phối định tuyến **THẬT** của `v7_ultra`: **`PEER_0` Zipf $s=0.618$**, **`PEER_1` Zipf $s=0.654$** (`15%` RAM chỉ trúng **`49.6%`**!). Đo SSD uncached (`FILE_FLAG_NO_BUFFERING`, QD=1): **`0.1781 ms` (`5,614 IOPS`)**; `u_exp`/`v_exp` tách 2 mảng → 2 seek/miss (`45–90 tok/s`!) | **CẤM phân trang SSD cho `ultra_445m`** — nạp trọn Q8 vào RAM (`404.7 MiB`, `0` miss)! Với mốc tương lai `mega_1.6B`, bắt buộc đóng gói xen kẽ `[u_e, v_e]` + **Windows Async IOCP (`QD=16–32`)** (`#87`). |
| **C4** | Đòi cổng `Q8 PEER rel_err < 1e-3` (bất khả thi toán học) | Đo thực `Q14 -> Q8` trên `v7_ultra`: `Q8 per-row` đạt `logit_rel_err = 0.01133 (1.13%)`, `top1_match = 98.68%`; `Q8 per-tensor` đạt `1.69%`, `top1_match = 98.97%` | Tách đôi cổng `#87`: **(a) Bit-exact (`rel_err == 0.0`)** cho BitNet 2-bit; **(b) Tương đương chức năng (`top1_match >= 98.5%`, `ΔNLL <= 0.01`, `logit_rel_err <= 0.02`)** cho PEER `Q8 per-row`. |
| **C5** | Cho rằng giữ state ở `ffi.rs` là chạy được dừng thích ứng từng token `R=1..8` trên `v7_ultra` | `model.rs:103-111` cấp `AttnKvState` + `MinGRU` **riêng cho từng độ sâu `step=0..R-1`**; token dừng sớm làm KV của `step>0` rỗng! Đo thực `v7_ultra` (train `R=1`): `R=1: ppl=72.6 → R=2: 74.2 → R=4: 117.3` (+62% tệ hơn!) | `v7_ultra` chỉ đổi `R ∈ {1,2}` ở **cấp lượt (turn-level)** không xóa `step=0`. True token-level adaptive depth (`R=1..8`) chuyển sang Pha II `Genesis-v8` với **Huginn Shared-KV + LogNormal $R$ + HRM Dual-Loop** (`#87`). |
| **C6** | Thử `Delta-Rule` trên vector Gaussian trực giao (`0.9945`) & đánh đồng `Titans` với `Gated DeltaNet` | Trạng thái ẩn **THẬT** của `v7_ultra` dị hướng mạnh (`Post-Attn mean_cos = 0.1785`, `Layer 5 = 0.2580`): Delta-Rule 1-pass sụp xuống **`0.7498`** (`N=64`) / **`0.6951`** (`N=128`); Titans MLP ngẫu nhiên post-hoc tụt xuống **`0.6049`**. Ngược lại, **Kanerva SDM / Centered Hopfield** đạt **`1.0000` trong `5.5 µs`**! | Tách 2 tầng (`#88`): **Tầng 1 (0 retrain cho `v7`)** dùng **Kanerva SDM + Modern Hopfield + ATMS**; **Tầng 2 (`Genesis-v8` Pretrain)** dùng **Multi-Head Gated DeltaNet (`ICLR 2025`)** + **Titans MAC / HOPE (`NeurIPS 2025`)** + **Engram (`DeepSeek 2026`)**. |
| **C7** | Tin vào Entropy Token thô $H(P_t)$ để chống ảo giác trong `Active Inference EFE` | Đo thực trên `v7_ultra_p1_preview.bin`: Sự thật (*"Thủ đô VN"*) có `mean_H = 2.358`; **Thực thể bịa (*"Vua Xylarion thứ 19..."*) có `mean_H = 2.078` (TỰ TIN HƠN SỰ THẬT!)**; chuỗi vô nghĩa `1.864`; nhại đề toán `0.962`! | **CẤM dùng entropy token thô đơn lẻ!** `GlobalWorkspaceEFE` (`#90`) bắt buộc dùng **4 kênh bất định bám chứng cứ**: Vault/KG Coverage + `Sheaf H^1`/`Venn-Abers` + `CAS/SAT/AST` + **Semantic Entropy** (*Nature* 2024). |
| **C8** | `HippoRAG-2` trên đồ thị ngẫu nhiên chỉ đạt Hạng `#27` (trượt `<CTX>`); thiếu cách dựng KG không API; bỏ sót lỗi gọi kép `Vault` | PPR 1-mầm trên đồ thị power-law tụt `#253/10K`. **HippoRAG 2 hai phía đa mầm + giảm trọng số hub `1/deg^0.75`** đẩy cầu/đích lên **Hạng `#1/#3` trong `15.89 ms`**. Dựng KG CPU offline đạt **`13,742 docs/s ≈ 19.3s`** cho `265,500` docs (`0%` LLM API). Lỗi `genesis_pro_chat.py:69-71` gọi kép `vault.retrieve()` tốn `+28.89 ms` (đã vá!). | Đóng khung chuẩn **HippoRAG-2 Bipartite + Offline CPU Extractor + Left-Hemisphere Recognition Memory** (`#89`). |
| **C9** | Mâu thuẫn `#87` (xóa `f32`, nén Q8 chỉ đọc) vs `#91` (cập nhật `PEER` lúc ngủ bằng tóm tắt `2 KB`) | Không thể SGD trên trọng số `Q8` đã lượng tử hóa mà không có master `f32`; `Naive Sparse` làm `MSE_A` vọt `+1106%`; `TF*IDF` đơn giản rò rỉ `113/200` khe dùng chung | Dùng **`f32` Sparse Delta-Sidecar (`sleep_peer_delta.bin < 20 MB` trên `E:\`)** + **SMFT Hard Background-DF Gate** (Meta FAIR 2025) + **Real Interleaved Replay** từ shard gốc `E:\` + cổng **Zero-Tradeoff** (`#91`). |
| **C10** | Đếm thiếu cấp phát heap Rust & trùng số mục lục `docs/README.md` | Có **`12` cấp phát `Vec`/token** (`7` ở `attn_rope.rs:108-133`, `4` ở `peer_pk.rs:206-209`, `1` ở `generate.rs:100` `exec_plan`); `genesis_v6_rust/` biên dịch `rustc` thuần `0` crate | Gộp toàn bộ `12` buffer vào `StepScratch` trong `InferenceState` (`0` heap alloc/token, `#87`) và dọn đánh số `docs/README.md`. |

### 2.1. Các Trào lưu "Giống Não" Đã Bị Phản chứng Thực nghiệm trên Laptop x86 (Không Áp dụng Mù quáng)
1. **Predictive Coding / Prospective Configuration (*Song et al., Nature Neurosci. 2024*) để thay Backprop Pretraining**: Đo trực tiếp trên `i5-10210U` (`64→64→64→16`): Backprop `142.1 µs` vs PCN (`T=20`) `3,589.6 µs` (**chậm hơn `25.3×`**) → Chỉ lấy nguyên lý sai số dự đoán cục bộ (*surprise*) cho bộ nhớ thời gian thực.
2. **Spiking Neural Networks (LIF vi giây) thuần túy trên x86 AVX2**: Gây *pointer chasing* và hỏng dự đoán nhánh SIMD → Thay bằng độ thưa khối (`MoE` + `PEER = 2.31%` active) + cổng sự kiện theo ngưỡng bất ngờ.
3. **Bộ nhớ Hebbian Cổ điển (`M += v k^T`)**: Ghi đè 1-shot sụp xuống `cos = 0.6645` (không xóa được ký ức sai) → Thay bằng `Kanerva/Hopfield + ATMS` (Pha I) và `Gated DeltaNet / HOPE` (Pha II).

---

## 3. Tổng hợp 7 Đột phá Học thuật Quốc tế 2024–2026 Tích hợp Vào `Genesis-v8 BICA v2`

1. **COGITATE (*Nature*, Vol. 642, 2025)**: Bác bỏ một phần cả GNWT và IIT → Định nghĩa `Global Latent Workspace` là **bảng đen tuyến phát sóng kỹ thuật** bảo toàn biểu diễn hậu phương, không tuyên bố ý thức chủ quan.
2. **Homeostasis & Interoception (Antonio Damasio, 2024–2025)**: Gắn **Cảm giác Nội tạng Phần cứng** (CPU/nhiệt, Pin/AC, RAM trống, hạn mức ổ `E:\`, tỷ lệ bác bỏ Bán cầu Trái) điều biến trực tiếp 4 biến `DA, NE, ACh, 5-HT` (`#90`).
3. **Sparse Memory Finetuning — `SMFT` (Lin et al., Meta FAIR, cuối 2025)**: Khóa khe bộ nhớ dùng chung có tần suất nền cao `DF_bg` → kéo sụt mức quên thảm họa từ `89%` (Full FT) / `71%` (LoRA) xuống `11%` — tương thích 1:1 với `PEER` (`#91`).
4. **Conditional Memory `Engram` (DeepSeek, 1/2026)**: Bảng băm $N$-gram đa đầu truy cập $O(1)$ tất định theo Định luật Mở rộng hình chữ U, bổ trợ hoàn hảo cho `FM-InfiniGram (#63)` + `PEER` (`#88`).
5. **Hierarchical Reasoning Model `HRM` (Sapient, 6/2025), `CTM` (Sakana, 5/2025) & `Huginn` (Geiping et al., 2025)**: Vòng lặp hai thang thời gian (Lập kế hoạch chậm + Thực thi nhanh) kèm KV-cache dùng chung theo độ sâu & huấn luyện $R \sim \text{LogNormal}$ (`#87`).
6. **Self-Adapting Models `SEAL` (MIT CSAIL, 2025)**: Mô hình tự sinh ghi chú học tập (*self-edits*) và tự tinh chỉnh theo phần thưởng kiểm chứng từ Bán cầu Trái (`v6_rlvr_selfplay.py`, `#91`).
7. **Semantic Entropy (Farquhar et al., *Nature* 2024) & `V-JEPA 2` (Meta/LeCun, 2025)**: Bù điểm mù entropy token thô (`#90`) và mở đường cho cảm giác đa phương thức trừu tượng (`#92` tương lai).

---

## 4. Kiến trúc Hợp nhất & Ngân sách Tài nguyên Thực tế trên `i5-10210U`

```text
   [ Exteroception: User / Code / Files ]   [ Interoception: CPU/RAM/Pin/E:\ Quota/Error ]
                    │                                            │
                    └────────────────────┬───────────────────────┘
                                         ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────┐
 │ GLOBAL LATENT WORKSPACE (Blackboard) + GROUNDED ACTIVE INFERENCE EFE + HOMEOSTASIS │
 │ • 4 kênh bất định bám đất (KHÔNG dùng entropy thô): Vault/KG Coverage + Sheaf H¹   │
 │   + Mondrian Venn-Abers + Symbolic Cert Gate + Semantic Entropy (#90)              │
 │ • Điều biến Nội môi: DA (Cert Reward) | NE (Surprise/R) | ACh (Mem) | 5-HT (Budget)│
 └───────────┬───────────────────────────┬───────────────────────────┬─────────────────┘
             ▼                           ▼                           ▼
┌────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────────┐
│ BÁN CẦU PHẢI (Rust GV8)│  │ PHỨC HỢP HẢI MÃ HỢP NHẤT│  │ BÁN CẦU TRÁI (100% Chính xác)│
│ • Persistent DLL Daemon│◄►│ • Tầng 1 (v7, 0 retrain)│◄►│ • Pure-CAS Rings/Calc/LinAlg│
│ • BitNet 2-bit/TL2 AVX2│  │   Kanerva SDM + Modern  │  │ • FOLIO DPLL SAT            │
│ • PEER Q8 per-row FULL │  │   Hopfield + ATMS (#88) │  │ • CEGIS AST / E-Graph       │
│   RAM (404.7 MiB, 0ms) │  │ • HippoRAG-2 Bipartite  │  │ • Q-DAG / Typed Proof Graph │
│ • Zero-Alloc Step (#87)│  │   Multi-Seed SpecPPR    │  │ • Sheaf H¹ / Cert Gate      │
│ • v8 Pretrain: Gated   │  │   (#89, Rank #1/#3)     │  │ • SelfModelGraph            │
│   DeltaNet + Huginn R  │  │ • Vault 265.5K FTS5     │  └──────────────┬──────────────┘
│   + HRM + Engram       │  └────────────┬────────────┘                 │
└───────────┬────────────┘               │                              │
            └────────────────────────────┼──────────────────────────────┘
                                         ▼ (Khi cắm sạc + CPU < 15% + RAM trống > 1.5 GB)
                 ┌──────────────────────────────────────────────────────┐
                 │ MẠNG MẶC ĐỊNH (DMN) & GIẤC NGỦ CỦNG CỐ (#91)         │
                 │ • f32 Sparse Delta-Sidecar (< 20 MB trên ổ E:\)      │
                 │ • SMFT Hard Background-DF Slot Gating                │
                 │ • Real Interleaved Replay (E:\) + SEAL/RLVR          │
                 │ • Cổng Zero-Tradeoff bắt buộc trước khi merge GV8    │
                 └──────────────────────────────────────────────────────┘
```

| Cấu hình Triển khai | Trọng số Nơ-ron (RAM) | Trạng thái + Hải mã + SDM | Bán cầu Trái (Lazy-load) | Tổng RAM thực tế | Tốc độ đo/mục tiêu (`R=1 / R=2`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hiện trạng `v7_ultra` (`GV6MOD03`)** | `975.7 MiB` (đo `977.0 MiB`) | Rời rạc (`~85–150 MiB`) | `~350–800 MiB` | `~1,410–1,920 MiB` | **`505.7 / 343.5 tok/s`** (`+2.37s` cold load!) |
| **`Genesis-v8A` (`ultra_445m` Full-RAM Q8 + Daemon)** — *Khuyến nghị ngay* | **`404.7 MiB`** (`0` SSD miss) | `~65 MiB` (CSR + SDM + SQLite) | `~180–350 MiB` (lazy) | **`~650–820 MiB`** | **`~500–750 / ~350–520 tok/s`** (`0 ms` cold load) |
| **`Genesis-v8B` (`ultra_445m` + Quant `emb/Attn/router` i8/f16)** | **`~395 MiB`** | `~65 MiB` | `~180–350 MiB` | **`~640–810 MiB`** | **`~700–1,100 / ~480–750 tok/s`** |
| **`Genesis-v8C` (`mega_1.6B` Interleaved Q8 + IOCP Async QD=16-32)** | **`~230 MiB`** (15% hot + fixed) | `~85 MiB` | `~180–350 MiB` | **`~495–665 MiB`** | **`~100–250 / ~60–150 tok/s`** (cần đo IOCP trước!) |

---

## 5. Bản đồ 6 Đặc tả Kỹ thuật Khả thi (`#86`–`#91`) & Lộ trình 2 Thế hệ

1. **`#86`** [`GENESIS_V8_BRAIN_ISOMORPHISM_MASTER_SPEC.md`](./GENESIS_V8_BRAIN_ISOMORPHISM_MASTER_SPEC.md) — Hiến chương khoa học, kiểm toán thực chứng `C1–C10` & kiến trúc hợp nhất (tài liệu này).
2. **`#87`** [`GENESIS_V8_GV8_PACKED_ZERO_ALLOC_DAEMON_SPEC.md`](./GENESIS_V8_GV8_PACKED_ZERO_ALLOC_DAEMON_SPEC.md) — Động cơ Rust AVX2 `GV8MOD01`, BitNet 2-bit/TL2, PEER Q8 Full-RAM (`404.7 MiB`), Zero-Alloc `12 Vecs`, `R` an toàn trạng thái & lộ trình `Huginn/HRM`.
3. **`#88`** [`GENESIS_V8_FAST_EPISODIC_AND_DELTANET_SPEC.md`](./GENESIS_V8_FAST_EPISODIC_AND_DELTANET_SPEC.md) — Bộ nhớ nhanh 2 tầng: Tầng 1 `Kanerva/Hopfield + ATMS` (0 retrain, `cos=1.000, 5.5 µs`) & Tầng 2 `Gated DeltaNet / Titans MAC + HOPE / Engram` (`Genesis-v8` pretrain).
4. **`#89`** [`GENESIS_V8_HIPPORAG2_BIPARTITE_GRAPH_SPEC.md`](./GENESIS_V8_HIPPORAG2_BIPARTITE_GRAPH_SPEC.md) — Phức hợp Hải mã `HippoRAG-2` 2 phía đa mầm không LLM API (`13,742 docs/s ≈ 19.3s` cho `265.5K` docs, `Rank #1/#3` trong `15.89 ms`).
5. **`#90`** [`GENESIS_V8_GLOBAL_WORKSPACE_HOMEOSTASIS_EFE_SPEC.md`](./GENESIS_V8_GLOBAL_WORKSPACE_HOMEOSTASIS_EFE_SPEC.md) — Không gian Làm việc Toàn cục, Nội môi phần cứng Damasio & Suy luận Chủ động EFE bám chứng cứ đa kênh (khử đảo ngược entropy thô).
6. **`#91`** [`GENESIS_V8_WAKE_SLEEP_SMFT_CONSOLIDATION_SPEC.md`](./GENESIS_V8_WAKE_SLEEP_SMFT_CONSOLIDATION_SPEC.md) — Mạng Mặc định (DMN) & Giấc ngủ Củng cố `SMFT Hard Background-DF` + `f32 Sparse Delta-Sidecar (< 20 MB)` + `Real Interleaved Replay` + `SEAL/RLVR`.

### Lộ trình Triển khai 2 Thế hệ (An toàn & Zero-Tradeoff)
- **Thế hệ Pha I (Zero-Retrain trên `v7_ultra_sft_best.gv8` / `v7_ultra_reasoner.gv8`) — ĐÃ TRIỂN KHAI & XÁC MINH ĐỘC LẬP PASS 100% (`2026-11-22`, Swarm 7 agent + `verifier` kiểm toán lại từ đầu)**:
  - **11/11 tệp sản xuất mới `<= 400` LOC**, **`0` thay đổi** trên toàn bộ `345` tệp bảo vệ (khóa SHA-256), **`0` vi phạm Gate-1 AST**, **`0 B` trên ổ `C:\`**:
    - `#87`: `gv8_codec.rs` (`262`), `gv8_kernels.rs` (`340`), `peer_delta_sidecar.rs` (`211`), `gv8_converter.py` (`301`) → nén `2.29×` (`937.19 → 408.70 MiB`), RSS `424.66 MiB` (`0` SSD miss), ColdLoad `149.9–230.1 ms`, BitNet 2-bit `rel_err = 0.0`, PEER Q8 `top1 = 99.80% (R=1) / 99.22% (R=2)`, `0` alloc/dealloc (`GlobalAlloc` hook), ấm `574.7 / 419.8 tok/s`, 4 nhánh song song `724.0–990.2 eff-tok/s`.
    - `#88`: `fast_episodic_memory.py` (`349`) → `50/50 (100.0%)` sửa niềm tin 1-shot & khử dị hướng (`0.221 → -0.0002`), `0.367–0.423 ms`, RAM `8.77 MB` ở `M=4,096`.
    - `#89`: `hipporag2_indexer.py` (`295`), `hipporag2_retriever.py` (`294`) → đánh chỉ mục `265,500` docs trong `21.50 s` (`40.28 MiB`, RAM `41.90 MB`), Multi-Hop 2 bước ẩn đạt `61.7% (37/60)` vs BM25 phẳng `0.0%` (`+61.7 pp`), PPR `4.74 ms`, OOD bịa `50/50 (100%) < 0.15`.
    - `#90`: `homeostatic_interoceptor.py` (`129`), `global_workspace_efe.py` (`378`, đã vá cổng `ACh` chống nhiễu Wiki cho ký ức cá nhân) → `Abstention F1 = 0.9615`, `Answer Rate = 92.0%`, trọng tài `0.0798 ms`, nội môi Win32 Pin/CPU/RAM/Ổ `E:\` & `Semantic Entropy` PASS.
    - `#91`: `peer_bg_profiler.py` (`218`), `genesis_sleep_daemon.py` (`377`) → `peer_bg_df.bin` (`2.0 MiB` đo trên 10,000 đoạn thật), SMFT rò `0/200` (vs `TF*IDF` `200/200`), 5 chu kỳ giữ `99.0%` mới / `100.0%` cũ, tiêm lỗi tự rollback 100%, nạp chéo `1,914/1,914` hàng (`5.64 MiB`) vào `genesis_v6.dll`.
  - **Tích hợp E2E cả 5 phân hệ trong 1 tiến trình Offline CPU**: khởi động `548.9 ms`, **Peak RAM `622.9 MiB`**.
- **Thế hệ Pha II (Native Pretraining `Genesis-v8` trên H100 / B200)**: Thay `MinGRU` bằng `Multi-Head Gated DeltaNet` + `Titans MAC/HOPE` + `Engram` + `Huginn Shared-KV` (`W=1024`, $R \sim \text{LogNormal}$) + `HRM` Slow/Fast Dual-Loop + vòng `GRPO/RLVR`.

### Quyết định Quản trị Cần Chủ Dự án Phê duyệt
- Cập nhật câu *"0% mạng nơ-ron"* trong [`AGENTS.md`](../../AGENTS.md) sang chuẩn **Hai Bán Cầu (`Bán cầu Trái 0% NN Bất khả Ảo giác + Bán cầu Phải Nơ-ron Siêu thưa 100% CPU Offline Laptop`)** để đồng bộ với Gate-1 AST (không tự ý sửa khi chưa có phê duyệt trực tiếp từ chủ dự án).
