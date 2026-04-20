[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/szn4ffc4)
# CSC4007 — Lab 2 Starter Kit (IMDB: Audit → Preprocess → Vectorize → Baseline ML)

Starter repo cho sinh viên, giữ nhịp làm việc gần với Lab 1 nhưng chuyển trọng tâm sang:
- audit dữ liệu văn bản,
- tiền xử lý vừa phải cho **IMDB review**,
- vector hoá bằng **BoW** hoặc **TF-IDF**,
- baseline ML bằng **Logistic Regression** hoặc **Linear SVM**,
- đánh giá bằng **macro-F1**, **accuracy**, **confusion matrix**,
- **error analysis** tối thiểu 10 mẫu sai.

Lab 1 tập trung vào hướng data-centric NLP & Data Card. Lab 2 nối tiếp sang pipeline **audit → preprocess → vectorize → baseline ML** với artefact chính là **bảng metric + confusion matrix + error analysis**.

## Chạy trên máy cá nhân
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python run_lab2.py --dataset imdb --seed 42 --vectorizer tfidf --model logreg
```

## Dataset mặc định
- **IMDB** từ Hugging Face `datasets`
- Repo vẫn giữ tùy chọn `local_csv` để giảng viên có thể đổi dữ liệu về sau, nhưng bài lab mặc định dùng **IMDB**.

## Cấu trúc repo
```text
csc4007_lab2_starter/
├── .github/workflows/ci.yml
├── data/raw/README.md
├── notebooks/README.md
├── reports/analysis_report.md
├── run_lab2.py
└── src/
    ├── audit_core.py
    ├── error_analysis.py
    ├── evaluate.py
    ├── load_data.py
    ├── modeling.py
    ├── preprocess.py
    ├── split.py
    └── utils.py
```

## CI làm gì?
- Cài dependencies.
- Chạy `run_lab2.py` với **IMDB** ở chế độ smoke test (`--max_rows` nhỏ).
- Kiểm tra các output bắt buộc tồn tại.

## Output bắt buộc sau khi chạy
- `outputs/logs/data_audit.md`
- `outputs/logs/audit_before.md`
- `outputs/logs/audit_after.md`
- `outputs/splits/train.csv`, `val.csv`, `test.csv`
- `outputs/metrics/metrics_summary.json`
- `outputs/metrics/metrics_summary.md`
- `outputs/figures/confusion_matrix.png`
- `outputs/error_analysis/error_analysis.csv`
- `outputs/error_analysis/error_analysis_summary.md`
- `outputs/pipeline/model_pipeline.joblib`
- `outputs/predictions/test_predictions.csv`

## Sinh viên cần làm tiếp
1. Chạy đúng với **IMDB**.
2. Chỉnh `src/preprocess.py` để thử ít nhất **2 chiến lược tiền xử lý**.
3. So sánh ít nhất **2 cấu hình** trong các nhóm sau:
   - raw vs cleaned text
   - BoW vs TF-IDF
   - Logistic Regression vs Linear SVM
   - unigram vs unigram+bigram
4. Phân tích **ít nhất 10 mẫu sai** trong `outputs/error_analysis/error_analysis.csv`.
5. Hoàn thành `reports/analysis_report.md`.

## Gợi ý thảo luận đúng với IMDB
- HTML tags, ký tự lặp (`!!!`), in hoa, số điểm kiểu “10/10” có nên xoá không?
- Giữ hay bỏ dấu câu có làm mất tín hiệu cảm xúc không?
- Trên IMDB, **accuracy** và **macro-F1** có chênh nhau nhiều không? Nếu dữ liệu lệch lớp thì chuyện gì xảy ra?
- Những lỗi nào khó hơn: phủ định, mỉa mai, review quá dài, hoặc sentiment trộn lẫn?
