"""
CS117 – Demo Thực Nghiệm (Trình bày cho Giảng Viên)
=====================================================
Script này:
    1. Sinh mock data lớn (30 căn hộ, 30 ngày)
    2. Chạy pipeline Feature Extraction
    3. Train RandomForest + LogisticRegression
    4. Đánh giá mô hình (Accuracy, F1, ROC-AUC, Confusion Matrix)
    5. Demo real-time: nhập apartment_id → dự đoán ngay

Chạy:  python demo_experiment.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # safe cho server/Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, sys, time, random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ─── Sklearn ─────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, roc_curve, f1_score)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn không có. Chạy: pip install scikit-learn")

np.random.seed(42); random.seed(42)
os.makedirs("output", exist_ok=True)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        CS117 – Nhận diện AirBnb lậu  │  DEMO THỰC NGHIỆM   ║
║        Feature Extraction + Classification Pipeline          ║
╚══════════════════════════════════════════════════════════════╝
"""
print(BANNER)


# ══════════════════════════════════════════════════
#  PHẦN 1: CẤU TRÚC DỮ LIỆU (giữ nguyên từ feature_extraction.py)
# ══════════════════════════════════════════════════

@dataclass
class ApartmentFeatureVector:
    apartment_id: str
    date: str
    total_unique_visitors: int = 0
    unknown_visitor_count: int = 0
    avg_visit_duration_min: float = 0.0
    max_visit_duration_min: float = 0.0
    luggage_event_count: int = 0
    luggage_visitor_ratio: float = 0.0
    peak_hour_event_ratio: float = 0.0
    night_entry_count: int = 0
    entries_per_day: float = 0.0
    visitor_turnover_rate: float = 0.0
    max_concurrent_visitors: int = 0
    label: Optional[int] = None
    profile: str = "normal"


# ══════════════════════════════════════════════════
#  PHẦN 2: MOCK DATA (30 căn hộ × 30 ngày, có noise)
# ══════════════════════════════════════════════════

FLOORS = list(range(1, 7))
UNITS  = list(range(1, 6))
APARTMENTS = [f"A{f*100 + u:03d}" for f in FLOORS for u in UNITS]

_rp = random.Random(99)  # đồng bộ với mock_data_generator.py
PROFILES = {}
for apt in APARTMENTS:
    r = _rp.random()
    PROFILES[apt] = "normal" if r < 0.60 else ("airbnb" if r < 0.80 else "suspect")

LABEL_MAP = {"normal": 0, "airbnb": 1, "suspect": 1}

def _clip(v, lo, hi): return float(np.clip(v, lo, hi))

def generate_feature_vector(apt_id: str, date: datetime, profile: str, seed: int):
    rng = np.random.default_rng(seed)
    dow = date.weekday()
    is_weekend = int(dow >= 4)
    md = date.day
    sb = 1.2 if md <= 10 else (0.85 if md >= 25 else 1.0)

    def noisy(v, sf=0.20):
        n = rng.normal(0, abs(v)*sf + 0.10)
        if rng.random() < 0.07: n *= rng.choice([2.8, 3.5, -2.0])
        return v + n

    if profile == "normal":
        bv, bur, bd, blr = rng.normal(2.5,1.3), rng.beta(2,6), rng.normal(38,18), rng.beta(1.5,7)
        bpr, bn, bt, mc  = rng.beta(2,4), rng.poisson(0.6), rng.beta(2,7), rng.integers(1,4)
    elif profile == "airbnb":
        bv  = rng.normal(6.5,2.1) * (1.3 if is_weekend else 1.0) * sb
        bur, bd, blr = rng.beta(7,2), rng.normal(18,8), rng.beta(6,3)
        bpr, bn, bt, mc = rng.beta(6,2), rng.poisson(2.5), rng.beta(8,2), rng.integers(2,6)
    else:
        bv  = rng.normal(4.0,1.5) * (1.1 if is_weekend else 1.0)
        bur, bd, blr = rng.beta(4,4), rng.normal(28,10), rng.beta(3,5)
        bpr, bn, bt, mc = rng.beta(4,4), rng.poisson(1.2), rng.beta(5,4), rng.integers(1,4)

    tv  = max(1, int(round(noisy(bv))))
    uc  = max(0, int(round(tv * _clip(noisy(bur,0.08), 0, 1))))
    ad  = _clip(noisy(bd, 0.15), 2, 240)
    md_ = _clip(ad + rng.uniform(5, 45), ad, 300)
    lr  = _clip(noisy(blr, 0.10), 0, 1)
    lc  = max(0, int(round(tv * lr)))
    pr  = _clip(noisy(bpr, 0.10), 0, 1)
    ne  = max(0, int(noisy(bn, 0.3)))
    epd = _clip(noisy(tv*1.1, 0.10), 0.5, 30)
    tr  = _clip(noisy(bt, 0.10), 0, 1)
    mcc = max(1, int(noisy(mc, 0.15)))

    return ApartmentFeatureVector(
        apartment_id=apt_id, date=date.strftime("%Y-%m-%d"),
        total_unique_visitors=tv, unknown_visitor_count=uc,
        avg_visit_duration_min=round(ad,2), max_visit_duration_min=round(md_,2),
        luggage_event_count=lc, luggage_visitor_ratio=round(lr,4),
        peak_hour_event_ratio=round(pr,4), night_entry_count=ne,
        entries_per_day=round(epd,4), visitor_turnover_rate=round(tr,4),
        max_concurrent_visitors=mcc,
        label=LABEL_MAP[profile], profile=profile
    )

print("─"*60)
print("📦 BƯỚC 1: Sinh mock data (30 căn hộ × 30 ngày) …")
START = datetime(2025, 5, 1)
records = []
for day_idx in range(30):
    d = START + timedelta(days=day_idx)
    for apt in APARTMENTS:
        seed = hash((apt, day_idx)) & 0xFFFFFFFF
        rec  = generate_feature_vector(apt, d, PROFILES[apt], seed)
        records.append(asdict(rec))

df = pd.DataFrame(records)

n0 = (df.label == 0).sum(); n1 = (df.label == 1).sum()
print(f"   ✅ {len(df)} samples  |  Normal={n0}  Violation={n1}  "
      f"(ratio {n1/len(df)*100:.1f}% vi phạm)")
df.to_csv("output/dataset_full.csv", index=False)
print("   💾 Đã lưu: output/dataset_full.csv")


# ══════════════════════════════════════════════════
#  PHẦN 3: TRAIN & EVALUATE MODELS
# ══════════════════════════════════════════════════

FEATURE_COLS = [
    "total_unique_visitors", "unknown_visitor_count",
    "avg_visit_duration_min", "max_visit_duration_min",
    "luggage_event_count",   "luggage_visitor_ratio",
    "peak_hour_event_ratio", "night_entry_count",
    "entries_per_day",       "visitor_turnover_rate",
    "max_concurrent_visitors"
]

print("\n─"*60)
print("🤖 BƯỚC 2: Huấn luyện mô hình phân loại …")

X = df[FEATURE_COLS].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                        max_depth=4, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=500, C=0.8, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_s if name == "Logistic Regression" else X_train
    Xte = X_test_s  if name == "Logistic Regression" else X_test

    t0 = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - t0

    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:,1]

    cv_scores = cross_val_score(model, Xtr if name != "Logistic Regression" else X_train_s,
                                 y_train, cv=5, scoring="f1", n_jobs=-1)

    results[name] = {
        "model":      model,
        "y_pred":     y_pred,
        "y_proba":    y_proba,
        "accuracy":   (y_pred == y_test).mean(),
        "f1":         f1_score(y_test, y_pred),
        "roc_auc":    roc_auc_score(y_test, y_proba),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std":  cv_scores.std(),
        "train_sec":  train_time,
    }
    print(f"   [{name:22s}]  Acc={results[name]['accuracy']:.3f}  "
          f"F1={results[name]['f1']:.3f}  AUC={results[name]['roc_auc']:.3f}  "
          f"CV-F1={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
          f"({train_time:.2f}s)")


# ══════════════════════════════════════════════════
#  PHẦN 4: VISUALIZATIONS
# ══════════════════════════════════════════════════

print("\n─"*60)
print("📊 BƯỚC 3: Sinh biểu đồ kết quả …")
sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = {0: "#4EACD1", 1: "#E8644B"}

# ── Fig A: Confusion matrices ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Bình thường","Vi phạm"],
                yticklabels=["Bình thường","Vi phạm"],
                linewidths=1, linecolor="white",
                annot_kws={"size": 13})
    ax.set_title(f"{name}\nAcc={res['accuracy']:.3f}  F1={res['f1']:.3f}", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrix – So sánh 3 mô hình", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("output/A_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ output/A_confusion_matrices.png")

# ── Fig B: ROC Curves ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
colors_roc = ["#E8644B", "#4EACD1", "#5CB85C"]
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, lw=2, color=col,
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
ax.fill_between([0,1],[0,1], alpha=0.04, color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve – So sánh 3 mô hình"); ax.legend(fontsize=10)
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("output/B_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ output/B_roc_curves.png")

# ── Fig C: Feature Importance (Random Forest) ─────────────────
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
colors_fi = ["#E8644B" if v > 0.10 else ("#F5A623" if v > 0.06 else "#AAD4E8")
             for v in importances.values]
importances.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
ax.axvline(0.06, color="orange", ls="--", lw=1.3, label="Ngưỡng trung bình")
ax.axvline(0.10, color="#E8644B", ls="--", lw=1.3, label="Ngưỡng quan trọng")
ax.set_xlabel("Gini Importance"); ax.legend(fontsize=9)
ax.set_title("Feature Importance – Random Forest", fontsize=13)
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig("output/C_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ output/C_feature_importance.png")

# ── Fig D: Cross-Validation F1 bar ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
names_cv  = list(results.keys())
means_cv  = [results[n]["cv_f1_mean"] for n in names_cv]
stds_cv   = [results[n]["cv_f1_std"]  for n in names_cv]
colors_cv = ["#E8644B", "#4EACD1", "#5CB85C"]
bars = ax.bar(names_cv, means_cv, color=colors_cv, alpha=0.8,
              edgecolor="white", width=0.5,
              yerr=stds_cv, capsize=5, error_kw={"elinewidth":2})
for bar, v in zip(bars, means_cv):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
            f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score (5-fold CV)")
ax.set_title("So sánh F1-Score (Cross Validation 5-fold)", fontsize=13)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("output/D_cv_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ output/D_cv_comparison.png")

# ── Fig E: Phân bố feature 2D scatter ────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for lbl, name in [(0, "Bình thường"), (1, "Vi phạm")]:
    sub = df[df.label == lbl]
    ax.scatter(sub["total_unique_visitors"], sub["visitor_turnover_rate"],
               c=PALETTE[lbl], alpha=0.2, s=12, label=name, rasterized=True)
ax.set_xlabel("Số khách riêng biệt / ngày", fontsize=11)
ax.set_ylabel("Visitor turnover rate",      fontsize=11)
ax.set_title("Phân bố 2D: Số khách vs Turnover Rate", fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/E_scatter_2d.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ output/E_scatter_2d.png")


# ══════════════════════════════════════════════════
#  PHẦN 5: TỔNG KẾT + DEMO DỰ ĐOÁN
# ══════════════════════════════════════════════════

print("\n─"*60)
print("📋 BƯỚC 4: Tổng kết kết quả")
print(f"\n{'Model':<22}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}  {'CV-F1':>12}  {'Time':>8}")
print("─"*70)
for name, res in results.items():
    print(f"{name:<22}  {res['accuracy']:>7.3f}  {res['f1']:>7.3f}  "
          f"{res['roc_auc']:>7.3f}  "
          f"{res['cv_f1_mean']:>5.3f}±{res['cv_f1_std']:.3f}  "
          f"{res['train_sec']:>7.2f}s")

best_name = max(results, key=lambda n: results[n]["roc_auc"])
best      = results[best_name]
print(f"\n🏆 Mô hình tốt nhất: {best_name}  (AUC = {best['roc_auc']:.3f})")
print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, best["y_pred"],
                             target_names=["Bình thường", "Vi phạm"]))


# ── Demo dự đoán 5 căn hộ ngẫu nhiên ─────────────────────────
print("─"*60)
print("🔍 BƯỚC 5: Demo dự đoán trên 5 căn hộ mẫu")
print()

best_model  = best["model"]
use_scaler  = (best_name == "Logistic Regression")

sample_apts = random.sample(APARTMENTS, 5)
demo_date   = datetime(2025, 5, 30)

print(f"{'Căn hộ':<8} {'GT':>6} {'Pred':>8} {'Prob Vi phạm':>14}  {'Đánh giá'}")
print("─"*60)

for apt_id in sample_apts:
    seed = hash((apt_id, 999)) & 0xFFFFFFFF
    vec  = generate_feature_vector(apt_id, demo_date, PROFILES[apt_id], seed)
    x    = np.array([[getattr(vec, c) for c in FEATURE_COLS]])
    if use_scaler:
        x = scaler.transform(x)
    prob = best_model.predict_proba(x)[0, 1]
    pred = int(prob >= 0.5)

    gt_str   = "Vi phạm" if vec.label == 1 else "Bình thường"
    pred_str = "Vi phạm" if pred      == 1 else "Bình thường"
    correct  = "✅" if pred == vec.label else "❌"
    risk     = "🔴 Cao" if prob > 0.70 else ("🟡 TB" if prob > 0.40 else "🟢 Thấp")

    print(f"{apt_id:<8} {gt_str:>10} {pred_str:>12}  {prob:>11.1%}  {risk}  {correct}")

print()
print("─"*60)
print("✅ Demo hoàn tất! Tất cả output lưu trong thư mục output/")
print("   output/dataset_full.csv")
print("   output/A_confusion_matrices.png")
print("   output/B_roc_curves.png")
print("   output/C_feature_importance.png")
print("   output/D_cv_comparison.png")
print("   output/E_scatter_2d.png")
