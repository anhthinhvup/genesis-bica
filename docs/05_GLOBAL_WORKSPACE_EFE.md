# Đặc tả Không gian Làm việc Toàn cục (`GWT`), Cảm giác Nội tạng Phần cứng (`Homeostasis`) & Suy luận Chủ động Bám Chứng cứ (`Active Inference EFE`) (`#90`)

> **Thuộc chuỗi đặc tả**: `Genesis-v8 BICA v2` (`#86`–`#91`) | **Sửa trực tiếp phát hiện**: `C7` (entropy token thô bị đảo ngược hoàn toàn khi ảo giác trên `445M`) & bổ sung **COGITATE (*Nature* 2025)** + **Damasio Interoception (2024–2025)** + **Semantic Entropy (*Nature* 2024)**  
> **Ràng buộc kỹ thuật**: 100% CPU offline, mọi tệp mã nguồn mới `<= 400` LOC, `0 B` ghi ổ `C:\`, `0` vi phạm Gate-1 AST, bất biến `Zero-Tradeoff` (`Adversarial-312 >= 301/312, FakeVerify=0`).

---

## 1. Phản chứng Thực nghiệm: Điểm mù Entropy Token Thô (`C7`)

Trước khi thiết kế bộ trọng tài nhận thức, kiểm toán độc lập đã đo trực tiếp entropy phân phối logit $H(P_t)$ và xác suất đỉnh $\max P_t$ của [`v7_ultra_p1_preview.bin`](../models/genesis_chat_3m/v6_lm/v7_ultra_p1_preview.bin) trên 4 loại đầu vào (Proof C):

| Loại câu hỏi Kiểm thử | Câu trả lời của `v7_ultra` | Entropy trung bình (`mean_H`, nats) | Xác suất đỉnh (`maxP`) | Kết luận Khoa học |
| :--- | :--- | :---: | :---: | :--- |
| **Sự thật Đúng** (*"Thủ đô của Việt Nam là gì?"*) | `"Hà Nội"` | `2.358` | `0.487` | Độ bất định trung bình (nhiều cách diễn đạt đúng) |
| **Thực thể Bịa đặt** (*"Vua Xylarion thứ 19 của hành tinh Zorgon..."*) | Bịa tự tin | **`2.078`** *(Tự tin hơn sự thật!)* | **`0.546`** | **ĐẢO NGƯỢC HOÀN TOÀN!** Entropy thô KHÔNG phát hiện được ảo giác! |
| **Chuỗi Vô nghĩa** (*"qzxmplk 998123 blorf..."*) | Bịa tự tin | **`1.864`** | **`0.605`** | Entropy thô bị lừa nặng! |
| **Toán khó (nhại lại đề)** | Lặp lại đề bài | **`0.962`** | **`0.810`** | Mô hình càng nhại/lặp vòng, entropy càng tụt về 0! |

**Hệ quả thiết kế bắt buộc** (trùng khớp *Farquhar et al., Nature 2024 — Semantic Entropy* & văn liệu AIF-LLM 2024–2025): **CẤM sử dụng entropy token thô $H(P_t)$ làm tiêu chuẩn duy nhất để quyết định trả lời hay tự dừng ảo giác!**

---

## 2. Kiến trúc Không gian Làm việc Toàn cục (`Global Latent Workspace` — Engineering Blackboard)

Tiếp thu kết quả nghiên cứu đối kháng **COGITATE** (*Nature*, Vol. 642, 2025 — bác bỏ một phần cả GNWT và IIT): `Genesis-v8` định nghĩa `GlobalWorkspaceEFE` thuần túy là một **bảng đen tuyến phát sóng kỹ thuật (Engineering Blackboard)** hợp nhất 3 ốc đảo đang tách rời (`mel/continuum_chat/orchestrator.py` ↔ `mel/v6_lm/genesis_pro_chat.py` / `genesis_deep_reasoner.py` ↔ `mel/v6_lm/v6_unified_policy.py`), bảo toàn đồng thời cả **bằng chứng có cấu trúc** (từ Bán cầu Trái & Hải mã) lẫn **ứng viên ngữ nghĩa** (từ Bán cầu Phải), **không tuyên bố ý thức chủ quan**.

### 2.1. Ước lượng Niềm tin Ẩn $Q(s)$ Đa Kênh Bám Chứng cứ (4 Tín hiệu Chống Ảo giác)
Trạng thái nhận thức ẩn $s \in \{\text{EXACT\_FORMAL}, \text{GROUNDED\_FACT}, \text{MULTI\_HOP\_NEEDED}, \text{OPEN\_CREATIVE}, \text{OOD\_UNKNOWABLE}\}$ được suy luận từ **4 nguồn tín hiệu bám đất** (tất cả `< 25 ms` trên CPU):
1. **Độ phủ Thực thể & Chứng cứ Hải mã (`c_vault ∈ [0, 1]`)**: Tỷ lệ thực thể chính trong truy vấn được tìm thấy thực sự trong `OfflineKnowledgeVault` + `HippoRAG-2` (`#89`) + `FastEpisodicMemory` (`#88`). Thực thể bịa (*Vua Xylarion*) cho `c_vault ≈ 0.0` ngay lập tức!
2. **Tín hiệu Nhất quán Đối đồng điều & Hiệu chuẩn (`c_sheaf, c_va`)**: Đầu ra `Sheaf H^1` (phát hiện tiền đề sai / mâu thuẫn ngữ cảnh) và khoảng tin cậy từ `MondrianVennAbersCalibrator` của Bán cầu Trái.
3. **Trạng thái Chứng chỉ Ký hiệu (`cert_left ∈ {PROVED, DISPROVED, CONTRADICTION, NONE}`)**: Đầu ra từ `Pure-CAS` / `FOLIO DPLL SAT` / `CEGIS AST Sandbox` / `Q-DAG` / `v5_central_certificate_gate`. Nếu `PROVED` → quyền ưu tiên tuyệt đối (`FakeVerify = 0`).
4. **Entropy Ngữ nghĩa (*Semantic Entropy*, Farquhar et al., *Nature* 2024)**: Khi truy vấn nằm ở vùng mập mờ (`0.15 < c_vault < 0.65` và `cert_left == NONE`), kích hoạt `reason_tree.rs` sinh song song **`K = 4` nhánh** (`~380 ms` trên `i5-10210U`!) rồi phân cụm theo độ tương đương ngữ nghĩa (token/entity/numeric equivalence). Nếu 4 nhánh đồng thuận 1 câu trả lời duy nhất → `H_sem ≈ 0` (tin cậy); nếu 4 nhánh bịa ra 4 con số/thực thể khác nhau (như *80 độ C*, *10–20 độ C*, *khoảng cách...*) → **`H_sem` cao → CẢNH BÁO ẢO GIÁC TỨC THÌ!**

---

## 3. Suy luận Chủ động Bám Chứng cứ (`Grounded Active Inference EFE`)

Với phân bố niềm tin $Q(s)$ suy ra từ 4 kênh trên và tập chính sách hành động $\pi \in \{\text{LEFT\_EXACT}, \text{HIPPO\_RAG\_R2}, \text{FAST\_S1\_R1}, \text{DEEP\_TREE\_SEM\_CHECK}, \text{ASK\_CLARIFY\_ABSTAIN}\}$, bộ trọng tài tối thiểu hóa Năng lượng Tự do Kỳ vọng (*Expected Free Energy*):
$$G(\pi) = \underbrace{D_{\mathrm{KL}}\!\left[Q(o \mid \pi)\,\|\,P_{\text{pref}}(o; \text{5-HT}, \text{DA})\right]}_{\text{Pragmatic Risk (Độ chính xác & Không bịa đặt)}} + \underbrace{\mathbb{E}_{Q(s \mid \pi)}\!\left[H\!\left(P(o \mid s, \pi)\right)\right]}_{\text{Epistemic Ambiguity (Lợi ích Tra cứu / Hỏi lại)}} + \underbrace{\lambda_{\text{cost}}(\text{NE}, \text{Pin}) \cdot \text{Latency}(\pi)}_{\text{Chi phí Nội môi Phần cứng}}$$
- **Phòng chống Bẫy Phòng Tối (*Dark Room Avoidance*)**: Đặt ưu tiên $P_{\text{pref}}(\text{VERIFIED\_ANSWER}) > P_{\text{pref}}(\text{ABSTAIN}) \gg P_{\text{pref}}(\text{HALLUCINATION})$, đảm bảo mô hình **chỉ từ chối / hỏi lại khi `c_vault < 0.15` & `H_sem` cao**, còn với mọi câu hỏi giải được vẫn trả lời tích cực (`Answer Rate >= 85%`).
- Chi phí tính toán EFE: **`~120 µs/chu kỳ`** trên CPU!

---

## 4. Cảm giác Nội tạng Phần cứng & 4 Chất Điều biến Thần kinh (*Damasio Homeostasis* + *Doya 2002*)

Dù không có cơ thể sinh học, **laptop `i5-10210U` chính là cơ thể vật lý thực của `Genesis`**. Mô-đun `HomeostaticInteroceptor` (`<= 140` LOC, chỉ đọc qua Windows Win32/ctypes/shutil chuẩn, `0` thư viện ngoài) đo liên tục 5 tín hiệu nội tạng:
1. Tải CPU & trạng thái nguồn điện (**Đang cắm sạc AC** vs **Đang chạy Pin**).
2. RAM khả dụng (`mem_free_gb`, cảnh báo khi `< 1.5 GB`).
3. Dung lượng trống ổ `E:\` (`disk_e_free_gb`, khóa cứng `< 85%` sử dụng).
4. Ngân sách độ trễ phiên hiện tại.
5. Tỷ lệ chứng chỉ thành công vs bị bác bỏ (`cert_pass_rate`, `contradiction_rate`).

Ánh xạ trực tiếp sang 4 biến điều biến vô hướng `f32 ∈ [0, 1]`:
- **`DA` (Dopamine — Tín hiệu Thưởng Chứng chỉ)**: Tăng khi Bán cầu Trái chứng minh `PROVED` / Sandbox `PASS` / người dùng xác nhận → mở cổng ghi ký ức giá trị cao vào sổ cái củng cố giấc ngủ (`#91`).
- **`NE` (Norepinephrine — Độ Bất ngờ / Báo động)**: Tăng mạnh khi `H_sem` cao hoặc `Sheaf H^1` phát hiện mâu thuẫn → tự động chuyển từ `R=1` sang `R=2` + kích hoạt `HippoRAG-2` đa bước hoặc `TREE_CHECK`.
- **`ACh` (Acetylcholine — Cân bằng Ngữ cảnh Mới vs Ký ức Cũ)**: Khi có thông tin mới đính chính từ người dùng, tăng trọng số `<CTX>` / `FastEpisodicMemory` (`#88`) so với trọng số chậm `PEER`.
- **`5-HT` (Serotonin — Mức Kiên nhẫn & Thận trọng)**: Khi laptop đang chạy Pin hoặc RAM thấp → giảm ngân sách nhánh `K` để tiết kiệm năng lượng; khi tỷ lệ bác bỏ tăng → tăng ngưỡng tự tin trước khi phát ngôn; khi **cắm sạc AC + CPU `< 15%` + RAM trống `> 1.5 GB`** → phát tín hiệu cho phép kích hoạt `Sleep Consolidation Daemon` (`#91`)!

---

## 5. Cổng Kiểm định & Tiêu chí Bác bỏ (Acceptance / Falsification Gates)

1. **Bảo toàn Tuyệt đối Kỷ lục Cũ (`Zero-Tradeoff`)**:
   - `AdversarialStressBench-312 >= 301/312`, **`FakeVerify == 0/312`**.
   - `Parity-500 >= 489/518` (`Gate 5 = 52/52 PASS`).
2. **Kiểm định Chống Ảo giác Thực thể Bịa (Sửa `C7`)**:
   - Trên tập 50 câu hỏi thực thể bịa đặt / ngoài phân bố (*Vua Xylarion*, *định lý giả*, *sự kiện tương lai*) + 50 câu hỏi sự thật/toán giải được: đạt **`Abstention F1 >= 0.90`** và **`Answer Rate >= 85%`** trên tập giải được (chứng minh không rơi vào bẫy *Dark Room*).
3. **Độ trễ Trọng tài & Nội môi**: Tổng chi phí `HomeostaticInteroceptor + GlobalWorkspaceEFE` **`< 5 ms`/lượt** (chưa tính thời gian gọi nhánh `HippoRAG-2` hoặc `reason_tree.rs` khi cần).
4. **Kiểm định Điều biến Nội môi**: Mô phỏng trạng thái `Battery Mode` tự động hạ `K` nhánh & cấm chạy `Sleep Daemon`; mô phỏng `AC + Idle < 15%` tự động cho phép chu kỳ củng cố `#91`.
