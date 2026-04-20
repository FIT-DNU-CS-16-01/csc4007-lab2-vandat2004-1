# Lab 2 Analysis Report (IMDB)

## 1. Data audit

### Số lượng mẫu
- **Tổng số mẫu**: 50,000 reviews từ IMDB dataset
- **Tất cả mẫu đều hợp lệ**: Không có dữ liệu thiếu (missing) hoặc rỗng (empty)

### Phân bố nhãn
- **Negative**: 25,000 (50%)
- **Positive**: 25,000 (50%)
- **Tỉ lệ cân bằng**: 1.0 (hoàn toàn cân bằng)

### Độ dài review điển hình
#### Theo ký tự:
- **Median**: 970 ký tự
- **P95 (percentile 95)**: 3,391 ký tự
- **Range**: 32 - 13,704 ký tự

#### Theo số từ:
- **Median**: 173 từ
- **P95**: 590 từ
- **Range**: 4 - 2,470 từ

### Missing / Empty text
- **Missing text**: 0 mẫu (0%)
- **Empty text**: 0 mẫu (0%)
- **Missing label**: 0 mẫu (0%)
- **Kết luận**: Dataset hoàn toàn sạch, không có giá trị thiếu

### Duplicates
- **Số mẫu trùng lặp chính xác**: 824 mẫu
- **Tỉ lệ**: 1.648% tổng dữ liệu
- **Ý nghĩa**: Một số review được đăng nhiều lần, cần xem xét loại bỏ để tránh rò rỉ dữ liệu giữa tập train/val/test

### 3 Quan sát đáng chú ý về dữ liệu IMDB

1. **Dataset hoàn toàn cân bằng và sạch**
   - 50-50 split giữa positive và negative, không có mất cân bằng lớp
   - Không có giá trị thiếu hoặc trống → dữ liệu chất lượng cao
   - Điều này làm cho accuracy và macro-F1 sẽ gần bằng nhau, không phản ánh chính xác hiệu suất thực tế trên dữ liệu lệch lớp

2. **Độ dài review rất đa dạng (32-13,704 ký tự)**
   - Median chỉ 970 ký tự nhưng max lên tới 13k → phân bố skewed (lệch phải)
   - Một số review rất ngắn (32 ký tự) có thể thiếu context để xác định sentiment
   - Một số review cực dài (>10k ký tự) có thể chứa nhiều ý kiến khác nhau, làm mô hình nhầm lẫn
   - TF-IDF weighted by term frequency có thể bị ảnh hưởng bởi sự khác biệt này

3. **Có duplicate cần xử lý**
   - 824 mẫu (1.648%) là bản sao chính xác của nhau
   - Nếu không xóa duplicate, có thể xảy ra data leakage: cùng review có thể xuất hiện cả ở train và test set
   - Cần loại bỏ hoặc một trong các bản sao để đảm bảo evaluation công bằng
   - Ảnh hưởng: Có thể làm cho metrics (F1, accuracy) cao hơn thực tế 1-2%

## 2. Preprocessing design

### 2.1 Các bước làm sạch được áp dụng
1. **Loại bỏ tags HTML**: Xóa các thẻ HTML như `<br>`, `<p>`, v.v. để lấy text tinh sạch
2. **Xử lý Unicode**: Loại bỏ các ký tự điều khiển không hợp lệ
3. **Thay thế URLs**: Thay `http://...` bằng token `<URL>` để giữ tín hiệu
4. **Thay thế emails**: Thay email patterns bằng `<EMAIL>`
5. **Chuyển thành chữ thường (lowercase)**: Chuẩn hóa toàn bộ text
6. **Xóa dấu câu**: Xóa hoàn toàn các ký tự đặc biệt như `!?.,;:()[]{}...`
7. **Chuẩn hóa khoảng trắng**: Loại bỏ khoảng trắng thừa

### 2.2 Xử lý dấu câu
**Quyết định: Xóa dấu câu hoàn toàn** (`drop_punct=True`)
- **Lý do**: Giảm kích thước feature space và tăng tốc độ vectorizer
- **Tradeoff**: Mất một số tín hiệu cảm xúc (ví dụ: "!" và "?" có thể chỉ mạnh độ cảm xúc)
- **Nhưng**: Đối với sentiment classification trên IMDB, từ vựng chính (từ mang cảm xúc) vẫn được giữ lại

### 2.3 Thay thế số
**Quyết định: KHÔNG thay số bằng `<NUM>`** (`replace_number=False`)
- **Lý do**: Các đánh giá của người dùng thường chứa số sao ("I give it 5 stars", "8 out of 10"), đây là tín hiệu cảm xúc quan trọng
- **Giữ lại số gốc**: Giúp mô hình học tập mối liên hệ giữa các con số và sentiment

### 2.4 Tín hiệu cảm xúc được cố ý giữ lại
- **Từ vựng**: Không loại bỏ stop words, giữ nguyên các từ mang cảm xúc mạnh mẽ (amazing, terrible, horrible, great)
- **Cấu trúc câu**: Giữ nguyên thứ tự từ để mô hình có thể học phủ định và nhấn mạnh
- **Ký tự lặp**: Không xóa các ký tự lặp (AAHHHHH, yessss) có thể chỉ mạnh độ cảm xúc

## 3. Experiment comparison

| Run | Text version | Vectorizer | Model | ngram | Macro-F1 | Accuracy | Ghi chú |
|---|---|---|---|---|---:|---:|---|
| 1 | cleaned | TF-IDF | LinearSVM | 2 | 0.9100 | 0.9100 | Baseline: drop_punct=True, replace_number=False, max_features=20k |

### Nhận xét:
- **Accuracy = Macro-F1 (0.91)**: IMDB dataset cân bằng hoàn toàn (50% negative, 50% positive) nên hai metric này giống nhau
- Điểm F1 chi tiết theo class:
  - **Negative**: precision=0.9184, recall=0.9000, f1=0.9091
  - **Positive**: precision=0.9020, recall=0.9200, f1=0.9109
- Pipeline này cho kết quả rất tốt trên tập test (5000 samples), khái quát hóa tốt từ training

## 4. Error analysis (>= 10 lỗi)

**Tổng số lỗi được export**: 450 mẫu (từ test set 5000)
**Số lỗi được chọn để phân tích**: 12 mẫu

### Phân loại lỗi: 3 nhóm chính

#### Nhóm 1: Phủ định / Cảm xúc trộn lẫn
- Mô hình bị nhầm vì review chứa phủ định ("not good", "hardly") nhưng vẫn có các từ tích cực
- Model confident cao (>99%) nhưng dự đoán sai

#### Nhóm 2: Review quá dài, chi tiết phức tạp
- Review dài (1000+ ký tự) với nhiều ý kiến khác nhau
- Mô hình bị lạc hướng bởi số lượng lớn tín hiệu mâu thuẫn

#### Nhóm 3: Sarcasm / Irony, bình luận tương đối
- Các review dùng mỉa mai hoặc có cấu trúc "tích cực nhưng thực chất tiêu cực"

### Bảng ghi lỗi chi tiết

| ID | True | Pred | Độ tin cậy | Nhóm lỗi | Giải thích ngắn |
|---|---|---|---|---|---|
| 31245 | NEG | POS | 0.9994 | Cảm xúc trộn lẫn | Review dài, khen rất nhiều diễn viên/phim khác => TF-IDF lấy quá nhiều từ tích cực |
| 37061 | NEG | POS | 0.9989 | Phủ định + sarcasm | "pure genius", "brilliant", nhưng thực tế là criticizing film => model bị đánh lừa |
| 8347 | NEG | POS | 0.9981 | Review tương đối | "excellent cast" ở đầu nhưng "unremarkable" ở cuối => Model ưu tiên từ đầu |
| 17596 | POS | NEG | 0.9931 | Phủ định | "not so well written", "not too terrible", "not that much better" => Phủ định chiếm ưu thế |
| 34543 | NEG | POS | 0.9910 | Review dài + phức tạp | Khen "incredibly", "great", "beautiful" nhưng overall là critical |
| 4648 | NEG | POS | 0.9906 | Cảm xúc trộn lẫn | Khen diễn viên ("so lovable", "funniest lines") nhưng cho rating 4/10 |
| 15598 | POS | NEG | 0.9735 | Phủ định | "imagine is terrible thing to waste" với sarcasm, nhưng model hiểu literal |
| 16142 | POS | NEG | 0.9712 | Phủ định | "direction is extremely boring", "story not interesting" => từ tiêu cực chiếm ưu thế |
| 32579 | NEG | POS | 0.9694 | Review tương đối | "expertly crafted", "intelligently directed" nhưng là "disturbing cannibal movie" |
| 14750 | POS | NEG | 0.9679 | Phủ định | "confusing", "muddled", nhưng "not as horrific" => so sánh confuses model |
| 41673 | POS | NEG | 0.9666 | Phủ định + review dài | "realistic", "great cinematography" nhưng "film is boring", "watching paint dry" |
| 46540 | POS | NEG | 0.9639 | Sarcasm + phủ định | "same thing over and over", "retarded" => mô hình hiểu "same thing" = tích cực |

### Nguyên nhân chính của lỗi:
1. **TF-IDF bị ảnh hưởng bởi từ vựng tích cực cục bộ** mặc dù overall sentiment là tiêu cực
2. **Mô hình không capture phủ định tốt**: "not good" được xem là 2 token riêng, TF-IDF 1-gram không hiểu ngữ cảnh
3. **Review dài gây confusion**: Nhiều topic, nhiều từ mâu thuẫn, mô hình lấy tổng thống kê thay vì hiểu ý chính
4. **Sarcasm và mỉa mai**: IMDB có rất nhiều review dùng irony, nhưng text-based model không thể capture điều này

### Đề xuất cải thiến:
- Sử dụng bigram features (`ngram_max=2`) để capture phủ định ("not good", "not bad")
- Thử dùng negation-aware preprocessing (thêm prefix `NOT_` trước các từ sau "not")
- Xem xét các giáo dục mô hình như transformer-based models (BERT) có thể hiểu context tốt hơn
- Tăng `max_features` để model có thêm capacity học các pattern phức tạp

## 5. Reflection

### 5.1 Pipeline tốt nhất trên IMDB
**Trả lời**: Pipeline **TF-IDF + LinearSVM** (với `ngram_max=2`, `drop_punct=True`, `replace_number=False`) đạt **F1 = 0.91** là kết quả rất tốt.

**Lý do**:
- TF-IDF là vector hóa đơn giản nhưng hiệu quả cho sentiment classification
- Bigram (`ngram_max=2`) giúp capture một số phủ định như "not good", "not bad"
- LinearSVM là classifier mạnh cho dữ liệu high-dimensional, học tốt trên TF-IDF vectors
- Bỏ dấu câu và không replace số giúp giữ tín hiệu đáng giá (rating numbers)
- IMDB là dataset khá sạch và balanced, nên các method tương đối đơn giản vẫn hoạt động tốt

### 5.2 Accuracy vs Macro-F1 trên IMDB
**Trả lời**: Không, chênh nhau rất ít (cả hai đều 0.91).

**Lý do**:
- IMDB dataset cân bằng hoàn toàn: 50% negative, 50% positive
- Khi dataset cân bằng, accuracy và macro-F1 tương đương
- Nếu dataset lệch lớp (ví dụ 90% positive, 10% negative), accuracy sẽ cao hơn macro-F1 rất nhiều

### 5.3 Metric nào phản ánh tốt hơn trên dataset lệch lớp
**Trả lời**: **Macro-F1** (hoặc weighted-F1) sẽ phản ánh tốt hơn độ khái quát hóa thực sự.

**Lý do chi tiết**:
- **Accuracy** sẽ bị lừa nếu model chỉ dự đoán lớp chiếm ưu thế (ví dụ dự đoán toàn "positive")
- Ví dụ: Dataset 90% positive, 10% negative → model chỉ cần dự đoán toàn "positive" đã đạt 90% accuracy, nhưng macro-F1 sẽ rất thấp (≈50%)
- **Macro-F1** xem xét cả precision và recall của từng lớp, không bị ảnh hưởng bởi sự mất cân bằng
- Đây là lý do tại sao trong machine learning thực tế, chúng ta luôn báo cáo macro-F1 khi dataset lệch lớp

### 5.4 Cải tiến muốn thử ở Lab 3
**Trả lời**: Có 3 hướng cải tiến chính:

1. **Negation Handling**: Thêm prefix `NOT_` vào các từ sau "not" (ví dụ "not good" → "NOT_good"). Điều này giúp mô hình phân biệt "good" từ "NOT_good"

2. **Transformer-based Models**: Thử BERT hoặc DistilBERT thay vì TF-IDF + LinearSVM. BERT hiểu context toàn diễn văn, có thể capture sarcasm, irony tốt hơn

3. **Advanced Text Features**: Kết hợp TF-IDF với các features khác:
   - Đếm từ cảm xúc tích cực/tiêu cực (lexicon-based)
   - Chiều dài review (người cho rating cao thường viết dài)
   - Tỷ lệ dấu câu cảm xúc (!!!, ???)

**Kỳ vọng**: Những cải tiến này có thể đưa F1 từ 0.91 lên 0.93-0.95, đặc biệt giúp giảm số lỗi trên các review có phủ định hoặc sarcasm.
