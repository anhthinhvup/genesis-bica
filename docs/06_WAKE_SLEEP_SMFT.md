# Đặc tả Mạng Mặc định (`DMN`) & Giấc ngủ Củng cố `SMFT + f32 Sparse Delta-Sidecar + SEAL/RLVR` (`#91`)

> **Thuộc chuỗi đặc tả**: `Genesis-v8 BICA v2` (`#86`–`#91`) | **Sửa trực tiếp phát hiện**: `C9` (xóa mâu thuẫn nội tại `#87 ↔ #91`, loại bỏ ngụy biện "tóm tắt 2 KB", vá lỗ hổng quên thảm họa `Naive Sparse +1106%` bằng **SMFT Hard Background-DF Gate**)  
> **Ràng buộc kỹ thuật**: 100% CPU offline, mọi tệp mã nguồn mới `<= 400` LOC, `0 B` ghi ổ `C:\`, bắt buộc qua cổng **Zero-Tradeoff** trước khi hợp nhất bất kỳ trọng số nào.

---

## 1. Giải quyết Triệt để Mâu thuẫn `#87 ↔ #91` & Bằng chứng Phản chứng (`C9`)

Báo cáo Round 1 từng tồn tại 3 mâu thuẫn kỹ thuật nghiêm trọng đã bị Round 2 phản chứng trực tiếp:
1. **Mâu thuẫn lượng tử hóa (`#87` vs `#91`)**: Nếu `#87` xóa toàn bộ trọng số `f32` và chỉ nạp `PEER Q8` (`bước 1/127`) chỉ đọc lúc chạy, mọi cập nhật SGD bước nhỏ ($\Delta w \sim 10^{-4}$) trong lúc ngủ sẽ bị làm tròn về `0` (vanishing quantized gradient)!
2. **Ngụy biện "tóm tắt thống kê 2 KB"**: Tóm tắt Gaussian `2 KB` không thể sinh văn bản ngôn ngữ/code thật cho LM — trong khi ổ `E:\` đang còn **`242.35 GB` trống** chứa sẵn toàn bộ kho shard thật và nhật ký phiên kiểm chứng!
3. **Thí nghiệm Verify 1 về Quên Thảm họa trên Khe Bộ nhớ Thưa**:
   - Cập nhật **Naive Sparse** vào các khe bộ nhớ đang kích hoạt làm sai số tác vụ cũ `MSE_A` **vọt tăng `+1,106%`** (quên thảm họa hoàn toàn vì chạm vào khe cú pháp dùng chung!).
   - Lọc theo trọng số `TF * IDF` đơn giản **vẫn để rò rỉ `113 / 200` khe dùng chung** nếu tần suất xuất hiện mới đủ cao!
   - Công trình **Sparse Memory Finetuning (`SMFT`, Lin et al., Meta FAIR, cuối 2025)** chứng minh giải pháp chính xác: **đóng băng cứng mọi khe có tần suất nền cao (`DF_bg > τ_bg`)**, kéo mức quên thảm họa từ `89%` (Full FT) / `71%` (LoRA) xuống chỉ còn **`11%`**!

---

## 2. Kiến trúc Ba Trụ cột Của `genesis_sleep_daemon`

### 2.1. Trụ cột 1 — Tệp Phụ Thưa `f32` (`sleep_peer_delta.bin < 20 MB` trên `E:\`)
Xóa hoàn toàn mâu thuẫn với `#87` (`GV8MOD01` giữ nguyên trạng thái chỉ đọc `404.7 MiB` nén `Q8` trong RAM!):
- Chỉ lưu một tệp phụ thưa nhỏ trên ổ `E:\` (`<WORKSPACE>\.cache_e\sleep_peer_delta.bin`, kích thước **`< 20 MB`**) gồm:
  - Danh sách chỉ số hàng chuyên gia `PEER` đang học (`active_expert_ids`, tối đa `1–5%` trong `262,144` chuyên gia = `2,600–13,000` hàng).
  - Bản sao độ chính xác cao **`delta_u[e], delta_v[e] ∈ f32[384]`** + trạng thái `Adafactor` cục bộ (`vu_local[e], vv_local[e]`) CHỈ cho các hàng này!
- Lúc chạy bình thường (Wake), `GV8MOD01` cộng trực tiếp `delta` cho các hàng nằm trong bảng băm nhỏ (hoặc hợp nhất lại thành hàng `Q8` mới sau khi vượt qua cổng Zero-Tradeoff). Trọng số lõi `BitNet` và `Attention` giữ **đóng băng 100%** khi ngủ trên CPU!

### 2.2. Trụ cột 2 — Khóa Khe Theo Tần suất Nền (`SMFT Hard Background-DF Gate`)
1. **Lập hồ sơ Tần suất Nền (`DF_bg`) Một Lần trên `v7_ultra_sft_best.bin`**:
   - Chạy suy luận không gradient qua `10,000` đoạn văn đa miền từ kho tiền huấn luyện trên ổ `E:\` (mất ~2 phút trên CPU).
   - Đếm số tài liệu kích hoạt mỗi chuyên gia $e \in \{0, \dots, 262{,}143\}$ ở cả `PEER_0` và `PEER_1`, lưu bảng `peer_bg_df.bin` (`2 × 262,144 × 4 B = 2.0 MB` trên `E:\`).
2. **Quy tắc Khóa Cứng (Hard Gate) Trước Mọi Bước Cập nhật Ngủ**:
   - Đóng băng tuyệt đối (**không bao giờ ghi**) tập chuyên gia cú pháp/nền chung:
     $$\mathcal{F}_{\text{frozen}} = \left\{ e \;\middle|\; \text{DF}_{\text{bg}}(e) > \tau_{\text{bg}} \right\}$$
     (ngưỡng mặc định $\tau_{\text{bg}}$ chọn tại phân vị 80% của phân bố `DF_bg` thực đo — đóng băng toàn bộ 20% chuyên gia phổ biến nhất chịu trách nhiệm ngữ pháp và kiến thức nền).
   - Chỉ chọn Top-$K_{\text{sleep}}$ chuyên gia đặc thù cao nhất nằm ngoài $\mathcal{F}_{\text{frozen}}$ theo tỷ số nâng:
     $$\text{Lift}(e) = \frac{\text{TF}_{\text{episode}}(e)}{\text{DF}_{\text{bg}}(e) + 1}$$

### 2.3. Trụ cột 3 — Trộn Tái phát Thật (`Real Interleaved Replay`) + `SEAL / RLVR Self-Edits`
Khi `HomeostaticInteroceptor` (`#90`) xác nhận điều kiện an toàn (**Cắm sạc AC + Tải CPU `< 15%` + RAM trống `> 1.5 GB` + Không có tương tác người dùng `>= 10 phút`**):
1. **Thu hoạch NREM (Episodic Harvesting)**: Đọc nhật ký phiên trong ngày từ `FastEpisodicMemory` (`#88`) chỉ chọn các tương tác có **tín hiệu thưởng `DA` cao** (được Bán cầu Trái xác chứng `PROVED`, Sandbox `PASS`, hoặc người dùng xác nhận sửa lỗi đúng).
2. **Sinh Ghi chú Tự học (`SEAL / RLVR`, MIT CSAIL 2025 + `v6_rlvr_selfplay.py`)**: Trong pha REM nhẹ, kích hoạt Bán cầu Trái (`Pure-CAS / FOLIO SAT / CEGIS AST`) tạo bài tập biến thể và tự chấm `PASS/FAIL` tuyệt đối, biến các ca giải đúng thành cặp huấn luyện `<THINK> ... </THINK>` sạch không nhiễu.
3. **Trộn Tái phát Thật (`Real Interleaved Replay`)**: Mỗi lô vi-củng cố trộn **`80%` mẫu thật từ kho shard gốc trên ổ `E:\` + `20%` mẫu mới đã xác chứng** (chống trôi phân phối mạnh hơn tuyệt đối so với tái phát giả lập `2 KB`).
4. **Hợp nhất Có Cổng `Zero-Tradeoff` (Atomic Commit / Auto-Rollback)**:
   - Sau chu kỳ củng cố (tối đa `50–200` bước trên các hàng `delta` không bị khóa), tự động chạy bộ kiểm định hồi quy nhanh:
     - `val_loss` held-out: $\Delta \mathcal{L}_{\text{val}} \le +0.005$.
     - `MC-8` & tập kiểm chứng cũ: không rớt bất kỳ câu nào đã đúng.
     - `Parity-500` & `Adversarial-312` (đối với lớp tích hợp): **`FakeVerify == 0`**.
   - **Nếu ĐẠT 100% cổng**: Ghi nguyên tử (`atomic rename`) `sleep_peer_delta.bin` và (tùy chọn) tái đóng gói hàng `Q8` cập nhật vào bản sao `GV8MOD01` mới.
   - **Nếu VI PHẠM BẤT KỲ ĐIỀU KIỆN NÀO**: Tự động hoàn nguyên (`rollback`) về `sleep_peer_delta.bin.bak` và ghi nhật ký nguyên nhân vào `<WORKSPACE>\.cache_e\sleep_audit.jsonl`.

---

## 3. Phân rã Mô-đun (`<= 400` LOC/Tệp)

- `mel/v6_lm/peer_bg_profiler.py` (`<= 220` LOC): Đo `DF_bg` thực trên `262,144 × 2` chuyên gia `PEER` từ `10,000` đoạn nền `E:\` và xuất `peer_bg_df.bin`.
- `mel/v6_lm/genesis_sleep_daemon.py` (`<= 380` LOC): Điều phối chu kỳ NREM/REM theo tín hiệu nội môi `#90`, quản lý `sleep_peer_delta.bin`, chạy SMFT Hard-Gate + Real Interleaved Replay + `v6_rlvr_selfplay.py`, và cưỡng chế cổng `Zero-Tradeoff`.
- `genesis_v6_rust/src/peer_delta_sidecar.rs` (`<= 260` LOC): Đọc/cộng/cập nhật thưa các hàng `f32` trong `sleep_peer_delta.bin` qua FFI không cần nạp lại `GV8MOD01`.

---

## 4. Cổng Kiểm định & Tiêu chí Bác bỏ (Acceptance / Falsification Gates)

1. **Chống Quên Thảm họa Sau 5 Chu kỳ Học Nối tiếp**:
   - Huấn luyện nối tiếp 5 gói tri thức mới (mỗi gói 20 sự kiện / quy tắc mới tách biệt) qua 5 chu kỳ ngủ:
     - Độ chính xác trên tri thức mới đạt **`>= 85%`** ngay cả khi tắt `<CTX>`.
     - Duy trì **`>= 98%`** năng lực trên `val_loss` nền, `MC-8`, `Parity-500` và `Adversarial-312` (`FakeVerify == 0`).
2. **Hiệu lực Khóa Cứng `SMFT`**: Kiểm chứng đối chứng chứng minh `SMFT Hard Background-DF Gate` giảm rò rỉ khe nền chung xuống **`0 / 200` khe** (so với `113 / 200` của `TF*IDF` thường và `+1,106%` `MSE_A` của `Naive Sparse`).
3. **An toàn Hệ thống & Ổ `C:\`**:
   - Tự động tạm dừng trong `< 1 giây` nếu người dùng quay lại gõ phím, rút sạc pin, hoặc CPU vượt `25%`.
   - Kích thước `sleep_peer_delta.bin <= 20 MB` trên ổ `E:\`, **`0 B` ghi lên ổ `C:\`**.
   - Kiểm thử tiêm lỗi (bơm dữ liệu nhiễu ác ý vào nhật ký ngày) bắt buộc bị cổng `Zero-Tradeoff` phát hiện và **hoàn nguyên tự động 100%**.
