# CS117 — Nhận diện căn hộ AirBnb lậu bằng Computer Vision

> Đồ án môn CS117 — Computational Thinking | UIT – ĐHQG TP.HCM | 2026

## Thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò |
|-----|-----------|------|---------|
| 1 | Nguyễn Đăng Khoa | 24520823 | Leader — Architect + Evaluation + Report |
| 2 | Nguyễn Duy Anh Khoa | 24520826 | Member — Feature Extraction |
| 3 | Nguyễn Đỗ Hoàng Khang | 24520754 | Member — Classifier (Rule + ML) |
| 4 | Võ Huy Khang | 24520772 | Member — Mock Data + EDA |

---

## Mô tả bài toán

Hệ thống tự động phát hiện căn hộ chung cư cho thuê ngắn hạn trái phép (AirBnb lậu) từ dữ liệu camera hành lang, dựa trên phân tích **hành vi nhóm** thay vì nhận dạng cá nhân — đảm bảo tuân thủ quyền riêng tư (Nghị định 13/2023/NĐ-CP).

**Input:** Event log từ camera hành lang (timestamp, apartment_id, has_luggage, is_known_resident...)  
**Output:** Danh sách căn hộ nghi vi phạm kèm risk score và video snippet bằng chứng

---

## Cấu trúc thư mục

```
cs117-airbnb-detection/
├── src/
│   ├── mock_data_generator.py   ← sinh dataset 900 mẫu + 8 EDA charts
│   └── demo.py                  ← pipeline train/eval + 5 ML charts
├── data/                        ← output của mock_data_generator.py
│   ├── dataset_full.csv         (900 × 18)
│   ├── dataset_summary.csv      (30 căn hộ)
│   ├── daily_trend.csv          (60 dòng)
│   └── charts/                  (8 PNG EDA)
├── output/                      ← output của demo.py
│   ├── dataset_full.csv
│   ├── A_confusion_matrices.png
│   ├── B_roc_curves.png
│   ├── C_feature_importance.png
│   ├── D_cv_comparison.png
│   └── E_scatter_2d.png
├── CS117_Colab.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cài đặt & Chạy

### Yêu cầu
- Python 3.9+
- Các thư viện trong `requirements.txt`

### Cài đặt

```bash
git clone https://github.com/[username]/cs117-airbnb-detection.git
cd cs117-airbnb-detection
pip install -r requirements.txt
```

### Chạy

**Bước 1 — Sinh mock dataset và EDA charts:**
```bash
python src/mock_data_generator.py
# Output: data/dataset_full.csv, data/charts/*.png
```

**Bước 2 — Train mô hình và xem kết quả:**
```bash
python src/demo.py
# Output: output/*.png, output/dataset_full.csv
```

**Hoặc chạy toàn bộ trên Google Colab:**  
Mở file `CS117_Colab.ipynb` → Run All

---

## Kết quả thực nghiệm

| Model | Accuracy | F1-Score | ROC-AUC | CV-F1 (5-fold) |
|-------|----------|----------|---------|----------------|
| Random Forest | 0.933 | 0.846 | 0.957 | 0.821 ± 0.037 |
| Gradient Boosting | 0.911 | 0.805 | 0.948 | 0.831 ± 0.030 |
| **Logistic Regression** | **0.928** | **0.847** | **0.964** | **0.857 ± 0.044** |

**Mô hình tốt nhất:** Logistic Regression (AUC = 0.964)  
**Top-3 features:** `luggage_visitor_ratio` > `visitor_turnover_rate` > `entries_per_day`

---

## Pipeline tổng thể

```
Video 24h → FrameExtractor (2 FPS) → YOLOv8 + ByteTrack → DOOR mapping
         → EventLog → FeatureExtractionEngine → ApartmentFeatureVector (12D)
                  ├─► RuleBasedClassifier  ──┐
                  └─► MLClassifier (RF/GB/LR) ─┴─► ViolationReport (JSON)
                                                        ▼
                                           Evidence Snippet + Alert
```

---

## Giảng viên hướng dẫn

TS. Ngô Đức Thành — Khoa Khoa học Máy tính, UIT – ĐHQG TP.HCM
