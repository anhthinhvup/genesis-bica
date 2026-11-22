# Đặc tả Phức hợp Hải mã Hai phía Đa mầm Không API (`HippoRAG-2` Offline CPU + Recognition Memory Bán cầu Trái) (`#89`)

> **Thuộc chuỗi đặc tả**: `Genesis-v8 BICA v2` (`#86`–`#91`) | **Sửa trực tiếp phát hiện**: `C8` (PPR 1-mầm trượt top-3 `<CTX>`, nút thắt OpenIE không dùng LLM API, lỗi gọi kép `vault.retrieve()` tốn `+28.89 ms`)  
> **Ràng buộc kỹ thuật**: 100% CPU offline, 0% gọi LLM/API ngoài, mọi tệp mã nguồn mới `<= 400` LOC, `0 B` ghi ổ `C:\`, `0` vi phạm Gate-1 AST.

---

## 1. Phân tích Nút thắt & Bằng chứng Thực nghiệm trên `knowledge_vault.sqlite` (`C8`)

Kho tri thức cục bộ `<WORKSPACE>\.cache_e\knowledge_vault.sqlite` hiện chứa **`265,500` tài liệu** (`150,000` bài Wikipedia Tiếng Việt + `115,500` mảnh `datasets_15gb`, `454.1 MB` trên đĩa `E:\`). Kiểm toán thực tế trên `i5-10210U` cho thấy 3 vấn đề gốc cần giải quyết:

1. **BM25 phẳng không giải được câu hỏi bắc cầu 2–3 bước (Multi-Hop)**: Nếu câu hỏi chứa thực thể $A$ và câu trả lời nằm ở đoạn chứa thực thể $C$ (nối qua cầu $B$), FTS5 BM25 hoàn toàn không khớp từ vựng ở đoạn $C$.
2. **PPR 1-mầm ngây thơ trượt khỏi cửa sổ `<CTX>`**:
   - Trên đồ thị ngẫu nhiên, PPR 1-mầm chỉ đưa đích 3-hop lên hạng `#27 / 20,000`.
   - Trên đồ thị 2 phía thực tế có nút trung tâm (*power-law hubs* như "Việt Nam", "năm 2020"), PPR 1-mầm không trọng số tụt xuống tận **Hạng `#253 / 10,000`** — vô dụng với cửa sổ `W=256` tokens (chỉ nạp được tối đa 2–3 đoạn `<CTX>`!).
3. **Nút thắt OpenIE không LLM API**: Bài báo gốc *HippoRAG 2* (*Gutierrez et al., 2025*) dùng GPT-4o-mini/Llama-3 OpenIE để trích xuất bộ ba `(s, r, o)` — **vi phạm nguyên tắc 100% CPU offline / 0% API** của `Genesis`!
4. **Lỗi gọi kép `Vault` trong mã nguồn cũ**: `mel/v6_lm/genesis_pro_chat.py:69-71` từng gọi `vault.retrieve()` rồi gọi tiếp `vault.format_ctx_prompt()` (gọi lại `retrieve()` lần thứ 2), lãng phí `+28.89 ms` (tổng `114.81 ms` trên truy vấn song ngữ dài) — **đã vá sạch**.

---

## 2. Kiến trúc `HippoRAG-2` Offline CPU Hai phía Đa mầm (`0%` LLM API)

### 2.1. Dựng Chỉ mục Đồ thị Hai phía `Passage ↔ Phrase` Hoàn toàn Trên CPU (`~19.3 giây` cho `265,500` tài liệu!)
Thay vì gọi LLM OpenIE, tận dụng cấu trúc tiêu đề, cụm danh từ viết hoa (Proper Noun Spans), cụm kỹ thuật và chỉ mục `concept_wsd.sqlite` sẵn có của `Genesis`:
- **Chứng minh thực nghiệm trên chính `knowledge_vault.sqlite` (`i5-10210U`)**: Bộ trích xuất cụm thực thể + tiêu đề thuần Python đạt tốc độ **`13,742 docs/s`** (`13.3` thực thể/tài liệu) → **đánh chỉ mục trọn bộ `265,500` tài liệu chỉ mất `~19.3 giây`** offline trên laptop!
- **Bố cục Lưu trữ Nén CSR trên ổ `E:\`** (`<WORKSPACE>\.cache_e\hipporag2_bipartite.bin`, `< 75 MB` RAM khi `mmap`/nạp):
  - **Tập nút đoạn văn** $V_P$ (`265,500` nút trỏ trực tiếp `doc_id` trong `knowledge_vault.sqlite`).
  - **Tập nút cụm thực thể / khái niệm** $V_E$ (lọc bỏ từ dừng và nút có bậc cực đại `df > 5,000`).
  - **Cạnh 2 phía có trọng số đặc hiệu**: Cạnh $(p, e)$ mang trọng số $w(p, e) = \text{tf}(e, p) \cdot \text{idf}(e)$, cộng thêm cạnh đồng nghĩa/khái niệm cục bộ từ `concept_wsd.sqlite` (WordNet / ViWN / Wikidata5M).

### 2.2. Thuật toán Truy hồi `Multi-Seed Hub-Suppressed Specificity PPR`
Khi nhận truy vấn $q$:
1. **Gieo mầm Kép (Dual Seeding — Đúng chuẩn *HippoRAG 2*, Gutierrez et al. 2025)**:
   - **Mầm Thực thể ($s_E$)**: Trích các cụm thực thể trong $q$, đối chiếu nút $e \in V_E$, gán trọng số đặc hiệu giảm phạt nút trung tâm:
     $$s_E(e) \propto \frac{1}{\deg(e)^{\gamma}}, \qquad \gamma = 0.75$$
   - **Mầm Đoạn văn ($s_P$)**: Lấy Top-15 đoạn văn ứng viên từ FTS5 BM25 (`viwiki`-first) của `OfflineKnowledgeVault`, chuẩn hóa điểm BM25 thành phân phối mầm $s_P$ với tỷ trọng pha trộn $\lambda_P = 0.35$.
   - Vector khởi tạo $p_q = (1 - \lambda_P)\hat{s}_E \oplus \lambda_P \hat{s}_P$.
2. **Lan truyền Đẳng hướng Giảm Hub trên Đồ thị CSR (`8–12` vòng lặp)**:
   $$r^{(k+1)} = \alpha \, p_q + (1 - \alpha)\, M_{\text{hub}}^\top r^{(k)}, \qquad \alpha = 0.50$$
   trong đó ma trận chuyển $M_{\text{hub}}$ chia bậc theo lũy thừa $\deg(u)^{-0.75}$ để chặn luồng xác suất bị hút vào các nút siêu phổ biến ("Việt Nam", "người", "năm").
   - **Bằng chứng thực nghiệm (Verify 3 trên đồ thị 2 phía power-law `10,000` nút ở `i5-10210U`)**: PPR 1-mầm tụt xuống Hạng **`#253`**, còn **Multi-Seed Hub-Suppressed PPR** đẩy đoạn cầu lên **Hạng `#1`** và đoạn đích 2-hop lên **Hạng `#3 / 10,000`** chỉ trong **`15.89 ms`**!

### 2.3. Bộ Lọc Ký ức Nhận diện (*Recognition Memory*) Bằng Bán cầu Trái (`0%` LLM API)
Trong *HippoRAG 2* gốc, bước lọc cuối dùng LLM đánh giá mức liên quan của các bộ ba ứng viên. Trong `Genesis-v8`, bước này được thay thế bằng cơ chế nhanh gấp `100×` và tất định 100% từ Bán cầu Trái:
1. Chấm tương hợp tiền đề & phản nghĩa bằng **`passage_span_sheaf_qa` (`Sheaf H^1` + `Daneš` progression)** trên Top-10 đoạn đầu ra của PPR (`< 8 ms`).
2. Hiệu chuẩn xác suất liên quan bằng **`MondrianVennAbersCalibrator`** để trả về **chỉ số độ phủ chứng cứ `evidence_coverage ∈ [0, 1]`** cho `GlobalWorkspaceEFE` (`#90`).
3. Chỉ bơm tối đa **2–3 đoạn văn sạch nhất (`<= 360–420` ký tự)** vừa khít cửa sổ `W=256` (`<CTX>`) — ưu tiên đoạn cầu + đoạn đích có liên kết thực thể chung.

---

## 3. Phân rã Mô-đun (`<= 400` LOC/Tệp)

- `mel/v6_lm/hipporag2_indexer.py` (`<= 300` LOC): Quét `knowledge_vault.sqlite`, trích cụm thực thể không regex/không API, xây và xuất `hipporag2_bipartite.bin` trên `E:\`.
- `mel/v6_lm/hipporag2_retriever.py` (`<= 320` LOC): Nạp CSR vào RAM, chạy Multi-Seed Hub-Suppressed PPR (`< 20 ms`), kết hợp Recognition Memory Bán cầu Trái và xuất `<CTX>` + `evidence_coverage`.
- *(Tùy chọn tăng tốc)*: Chuyển vòng lặp SpMV CSR 12 bước sang 1 hàm FFI `< 80` dòng trong `genesis_v6_rust` nếu đồ thị vượt `1M` nút (`< 3 ms`/truy vấn).

---

## 4. Cổng Kiểm định & Tiêu chí Bác bỏ (Acceptance / Falsification Gates)

1. **Hiệu năng Lập chỉ mục Offline**: Lập chỉ mục đầy đủ `265,500` tài liệu `knowledge_vault.sqlite` trên `i5-10210U` trong **`< 45 giây`**, `0%` API ngoài, `0 B` ghi ổ `C:\`.
2. **Chất lượng Bắc cầu Đa bước (Multi-Hop)**: Trên tập kiểm thử 100 câu hỏi bắc cầu 2–3 bước (đối chiếu Wikipedia VN + STEM không trùng từ khóa ở đoạn đích), **Recall@3 vượt FTS5 BM25 phẳng ít nhất `+20 pp`** (đủ lọt top-3 `<CTX>` cho `W=256`).
3. **Độ trễ & RAM**: Thời gian chạy PPR **`< 20 ms`**, tổng độ trễ truy hồi (FTS5 + PPR + Recognition Memory) **`< 60 ms`**; RAM đồ thị CSR **`< 80 MB`**.
4. **Độ trung thực Tín hiệu Bất định**: Trên 50 câu hỏi thực thể bịa đặt (*Vua Xylarion*, *hành tinh Zorgon*...), `evidence_coverage` trả về **`< 0.15`** ở `>= 95%` số ca để kích hoạt đúng nhánh `ABSTAIN / ASK_CLARIFY` trong `#90`.
