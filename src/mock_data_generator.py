"""
CS117 – Mock Data Generator (Báo cáo)
======================================
Sinh dataset lớn: 30 căn hộ × 30 ngày = ~900 feature vectors
Có noise thực tế, phân phối hành vi hợp lý cho biểu đồ báo cáo.

Output:
    data/dataset_full.csv        ← toàn bộ feature vectors
    data/dataset_summary.csv     ← tổng hợp theo căn hộ
    data/daily_trend.csv         ← xu hướng theo ngày
    data/charts/                 ← tất cả chart PNG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, random, json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

np.random.seed(42)
random.seed(42)

os.makedirs("data/charts", exist_ok=True)

# ══════════════════════════════════════════════════
#  CẤU HÌNH CHUNG CƯ (30 phòng)
# ══════════════════════════════════════════════════

FLOORS = [1, 2, 3, 4, 5, 6]
UNITS  = [1, 2, 3, 4, 5]

APARTMENTS = [f"A{floor*100 + unit:03d}"
              for floor in FLOORS for unit in UNITS]   # A101 … A605

# Phân loại thực tế (ground truth)
#   "normal"   : cư dân thường
#   "airbnb"   : cho thuê AirBnb (vi phạm)
#   "suspect"  : nghi ngờ nhẹ (biên giới)
APARTMENT_PROFILES = {}
rng_profile = random.Random(99)
for apt in APARTMENTS:
    r = rng_profile.random()
    if r < 0.60:
        APARTMENT_PROFILES[apt] = "normal"
    elif r < 0.80:
        APARTMENT_PROFILES[apt] = "airbnb"
    else:
        APARTMENT_PROFILES[apt] = "suspect"

# Số label: 0 = bình thường, 1 = vi phạm (airbnb + suspect)
LABEL_MAP = {"normal": 0, "airbnb": 1, "suspect": 1}

LABEL_COUNT = {0: sum(1 for v in APARTMENT_PROFILES.values() if LABEL_MAP[v]==0),
               1: sum(1 for v in APARTMENT_PROFILES.values() if LABEL_MAP[v]==1)}
print(f"[INFO] Apartments: {len(APARTMENTS)} total | "
      f"Normal={LABEL_COUNT[0]}, Violation={LABEL_COUNT[1]}")


# ══════════════════════════════════════════════════
#  SINH FEATURE VECTOR CÓ NOISE
# ══════════════════════════════════════════════════

def clip(val, lo, hi):
    return float(np.clip(val, lo, hi))

def generate_vector(apt_id: str, date: datetime,
                    profile: str, day_seed: int) -> dict:
    """
    Sinh 1 feature vector cho 1 căn hộ trong 1 ngày.
    Mỗi feature được sinh từ phân phối thống kê phù hợp với loại phòng.
    Có noise Gaussian + outlier thỉnh thoảng để dữ liệu thực tế hơn.
    """
    rng = np.random.default_rng(day_seed)

    # ── Ngày trong tuần / tháng ảnh hưởng hành vi ──────────────
    dow = date.weekday()           # 0=Mon … 6=Sun
    is_weekend = dow >= 4          # Fri, Sat, Sun
    month_day  = date.day
    season_boost = 1.2 if month_day <= 10 else (0.85 if month_day >= 25 else 1.0)

    # ── Tham số theo profile ────────────────────────────────────
    if profile == "normal":
        base_visitors      = rng.normal(2.1, 0.7)
        base_unknown_ratio = rng.beta(1.5, 8)      # mostly known residents
        base_duration      = rng.normal(42, 12)    # long stays (lives there)
        base_luggage_ratio = rng.beta(1, 12)       # rare luggage
        base_peak_ratio    = rng.beta(2, 5)
        base_night         = int(rng.poisson(0.3))
        base_turnover      = rng.beta(1, 9)
        max_conc           = rng.integers(1, 3)
    elif profile == "airbnb":
        base_visitors      = rng.normal(6.5, 2.1) * (1.3 if is_weekend else 1.0) * season_boost
        base_unknown_ratio = rng.beta(7, 2)        # mostly strangers
        base_duration      = rng.normal(18, 8)     # short stays
        base_luggage_ratio = rng.beta(6, 3)        # lots of luggage
        base_peak_ratio    = rng.beta(6, 2)        # peak hour check-in/out
        base_night         = int(rng.poisson(2.5))
        base_turnover      = rng.beta(8, 2)
        max_conc           = rng.integers(2, 6)
    else:  # suspect
        base_visitors      = rng.normal(4.0, 1.5) * (1.1 if is_weekend else 1.0)
        base_unknown_ratio = rng.beta(4, 4)
        base_duration      = rng.normal(28, 10)
        base_luggage_ratio = rng.beta(3, 5)
        base_peak_ratio    = rng.beta(4, 4)
        base_night         = int(rng.poisson(1.2))
        base_turnover      = rng.beta(5, 4)
        max_conc           = rng.integers(1, 4)

    # ── Noise ngẫu nhiên (5% outlier cứng) ─────────────────────
    def noisy(val, sigma_frac=0.12):
        noise = rng.normal(0, abs(val) * sigma_frac + 0.01)
        if rng.random() < 0.05:                    # 5% outlier mạnh
            noise *= rng.choice([2.5, 3.0, -1.5])
        return val + noise

    total_visitors   = max(1, int(round(noisy(base_visitors))))
    unknown_count    = max(0, int(round(total_visitors * clip(noisy(base_unknown_ratio,0.08), 0, 1))))
    avg_dur          = clip(noisy(base_duration, 0.15), 2, 240)
    max_dur          = clip(avg_dur + rng.uniform(5, 45), avg_dur, 300)
    luggage_ratio    = clip(noisy(base_luggage_ratio, 0.10), 0, 1)
    luggage_count    = max(0, int(round(total_visitors * luggage_ratio)))
    peak_ratio       = clip(noisy(base_peak_ratio, 0.10), 0, 1)
    night_entries    = max(0, int(noisy(base_night, 0.3)))
    entries_per_day  = clip(noisy(total_visitors * 1.1, 0.10), 0.5, 30)
    turnover         = clip(noisy(base_turnover, 0.10), 0, 1)
    max_concurrent   = max(1, int(noisy(max_conc, 0.15)))

    return {
        "apartment_id":          apt_id,
        "date":                  date.strftime("%Y-%m-%d"),
        "profile":               profile,
        "label":                 LABEL_MAP[profile],
        "floor":                 int(apt_id[1]),
        "unit":                  int(apt_id[2:]),
        "is_weekend":            int(is_weekend),
        # ── Group 1: Tần suất ──
        "total_unique_visitors": total_visitors,
        "unknown_visitor_count": unknown_count,
        "avg_visit_duration_min": round(avg_dur, 2),
        "max_visit_duration_min": round(max_dur, 2),
        # ── Group 2: Hành lý ──
        "luggage_event_count":   luggage_count,
        "luggage_visitor_ratio": round(luggage_ratio, 4),
        # ── Group 3: Thời gian ──
        "peak_hour_event_ratio": round(peak_ratio, 4),
        "night_entry_count":     night_entries,
        "entries_per_day":       round(entries_per_day, 4),
        # ── Group 4: Đa dạng ──
        "visitor_turnover_rate": round(turnover, 4),
        "max_concurrent_visitors": max_concurrent,
    }


# ══════════════════════════════════════════════════
#  SINH TOÀN BỘ DATASET (30 căn hộ × 30 ngày)
# ══════════════════════════════════════════════════

START_DATE = datetime(2025, 5, 1)
N_DAYS     = 30

rows = []
for day_idx in range(N_DAYS):
    date = START_DATE + timedelta(days=day_idx)
    for apt in APARTMENTS:
        seed = hash((apt, day_idx)) & 0xFFFFFFFF
        row  = generate_vector(apt, date, APARTMENT_PROFILES[apt], seed)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/dataset_full.csv", index=False)
print(f"[INFO] dataset_full.csv → {len(df)} rows × {len(df.columns)} cols")


# ── Tổng hợp theo căn hộ ─────────────────────────────────────
summary_cols = [
    "total_unique_visitors", "unknown_visitor_count",
    "avg_visit_duration_min", "luggage_visitor_ratio",
    "peak_hour_event_ratio", "night_entry_count",
    "visitor_turnover_rate", "max_concurrent_visitors"
]
df_summary = (
    df.groupby(["apartment_id", "profile", "label", "floor", "unit"])[summary_cols]
    .mean()
    .round(3)
    .reset_index()
)
df_summary.to_csv("data/dataset_summary.csv", index=False)
print(f"[INFO] dataset_summary.csv → {len(df_summary)} rows")

# ── Xu hướng theo ngày ───────────────────────────────────────
df_daily = (
    df.groupby(["date", "label"])[summary_cols]
    .mean()
    .round(3)
    .reset_index()
)
df_daily.to_csv("data/daily_trend.csv", index=False)
print(f"[INFO] daily_trend.csv → {len(df_daily)} rows")


# ══════════════════════════════════════════════════
#  CHARTING – tất cả chart dùng cho báo cáo
# ══════════════════════════════════════════════════

PALETTE = {
    "normal":  "#4EACD1",
    "airbnb":  "#E8644B",
    "suspect": "#F5A623",
    0:         "#4EACD1",
    1:         "#E8644B",
}
sns.set_theme(style="whitegrid", font_scale=1.05)
LABEL_NAMES = {0: "Bình thường", 1: "Vi phạm (AirBnb)"}

# ─── Chart 1: Phân bố nhãn (Pie) ────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
counts = [LABEL_COUNT[0], LABEL_COUNT[1]]
labels_pie = [f"Bình thường\n(n={LABEL_COUNT[0]})", f"Vi phạm\n(n={LABEL_COUNT[1]})"]
colors_pie = [PALETTE[0], PALETTE[1]]
wedges, texts, autotexts = ax.pie(
    counts, labels=labels_pie, colors=colors_pie,
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(linewidth=2, edgecolor="white"),
    textprops=dict(fontsize=12)
)
for a in autotexts:
    a.set_fontsize(13); a.set_fontweight("bold")
ax.set_title("Phân bố nhãn trong dataset\n(30 căn hộ × 30 ngày)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("data/charts/01_label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 01_label_distribution.png")

# ─── Chart 2: Violin – total_unique_visitors theo nhãn ──────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
features_violin = [
    ("total_unique_visitors",  "Số lượt khách riêng biệt / ngày"),
    ("luggage_visitor_ratio",  "Tỉ lệ người có hành lý"),
]
for ax, (feat, title) in zip(axes, features_violin):
    data_0 = df[df.label == 0][feat]
    data_1 = df[df.label == 1][feat]
    parts = ax.violinplot([data_0, data_1], positions=[0, 1],
                          showmedians=True, showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], [PALETTE[0], PALETTE[1]])):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Bình thường", "Vi phạm"], fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(feat)
    ax.grid(axis="y", alpha=0.4)
plt.suptitle("Violin Plot – So sánh đặc trưng giữa 2 nhóm", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("data/charts/02_violin_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 02_violin_features.png")

# ─── Chart 3: Box plot – 4 feature quan trọng ───────────────
key_features = [
    "total_unique_visitors", "visitor_turnover_rate",
    "peak_hour_event_ratio",  "night_entry_count"
]
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, feat in zip(axes, key_features):
    data_groups = [df[df.label == lbl][feat].values for lbl in [0, 1]]
    bp = ax.boxplot(data_groups, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="gray",
                                    markersize=3, alpha=0.5),
                    widths=0.5)
    for patch, col in zip(bp["boxes"], [PALETTE[0], PALETTE[1]]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Normal", "Violation"], fontsize=10)
    ax.set_title(feat.replace("_", "\n"), fontsize=9, pad=6)
    ax.grid(axis="y", alpha=0.4)
plt.suptitle("Box Plot – Top 4 Feature Phân biệt", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("data/charts/03_boxplot_top4.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 03_boxplot_top4.png")

# ─── Chart 4: Correlation heatmap ───────────────────────────
num_cols = [
    "total_unique_visitors", "unknown_visitor_count",
    "avg_visit_duration_min", "luggage_visitor_ratio",
    "peak_hour_event_ratio", "night_entry_count",
    "visitor_turnover_rate", "max_concurrent_visitors", "label"
]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 9},
            cbar_kws={"shrink": 0.8})
ax.set_title("Ma trận tương quan giữa các feature và nhãn", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("data/charts/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 04_correlation_heatmap.png")

# ─── Chart 5: Xu hướng theo ngày (Line) ─────────────────────
df_daily["date_dt"] = pd.to_datetime(df_daily["date"])
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
trend_features = [
    ("total_unique_visitors",  "Khách TB / ngày"),
    ("luggage_visitor_ratio",  "Tỉ lệ người có hành lý"),
    ("peak_hour_event_ratio",  "Tỉ lệ sự kiện giờ cao điểm"),
    ("night_entry_count",      "Số lượt vào ban đêm (>22h)"),
]
for ax, (feat, ylabel) in zip(axes.flat, trend_features):
    for lbl in [0, 1]:
        sub = df_daily[df_daily.label == lbl].sort_values("date_dt")
        ax.plot(sub["date_dt"], sub[feat],
                color=PALETTE[lbl], linewidth=2,
                label=LABEL_NAMES[lbl], alpha=0.9)
        ax.fill_between(sub["date_dt"], sub[feat],
                         alpha=0.08, color=PALETTE[lbl])
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
plt.suptitle("Xu hướng theo thời gian – Normal vs Violation (30 ngày)", fontsize=14)
plt.tight_layout()
plt.savefig("data/charts/05_daily_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 05_daily_trend.png")

# ─── Chart 6: Scatter – Visitor vs Turnover ─────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for lbl in [0, 1]:
    sub = df[df.label == lbl]
    ax.scatter(sub["total_unique_visitors"], sub["visitor_turnover_rate"],
               c=PALETTE[lbl], alpha=0.25, s=15, label=LABEL_NAMES[lbl])
# Convex-hull vùng vi phạm gần đúng (annotation)
ax.annotate("Vùng nghi AirBnb\n(nhiều khách, turnover cao)",
            xy=(9, 0.85), fontsize=9, color=PALETTE[1],
            arrowprops=dict(arrowstyle="->", color=PALETTE[1]),
            xytext=(11, 0.65))
ax.set_xlabel("Số khách riêng biệt / ngày", fontsize=11)
ax.set_ylabel("Visitor turnover rate", fontsize=11)
ax.set_title("Scatter: Số khách vs Tỉ lệ thay thế khách", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/charts/06_scatter_visitor_turnover.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 06_scatter_visitor_turnover.png")

# ─── Chart 7: Feature importance (Pearson |r| with label) ───
feat_importance = df[num_cols].corrwith(df["label"]).abs().drop("label").sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = [PALETTE[1] if v > 0.4 else ("#F5A623" if v > 0.25 else "#AAD4E8")
              for v in feat_importance.values]
feat_importance.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="white")
ax.axvline(0.25, color="orange", linestyle="--", linewidth=1.2, label="Ngưỡng trung bình")
ax.axvline(0.40, color=PALETTE[1], linestyle="--", linewidth=1.2, label="Ngưỡng cao")
ax.set_xlabel("|Pearson r| với nhãn vi phạm", fontsize=11)
ax.set_title("Mức độ tương quan của từng feature với nhãn", fontsize=13)
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig("data/charts/07_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 07_feature_importance.png")

# ─── Chart 8: Heatmap theo tầng / unit ──────────────────────
pivot = df_summary.pivot_table(index="floor", columns="unit",
                               values="total_unique_visitors", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"Unit {c}" for c in pivot.columns], fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"Tầng {r}" for r in pivot.index], fontsize=10)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        apt_id = f"A{list(pivot.index)[i]*100 + list(pivot.columns)[j]:03d}"
        lbl = APARTMENT_PROFILES.get(apt_id, "normal")
        marker = "★" if lbl == "airbnb" else ("▲" if lbl == "suspect" else "")
        ax.text(j, i, f"{val:.1f}\n{marker}", ha="center", va="center",
                fontsize=8, color="black")
plt.colorbar(im, ax=ax, label="TB khách / ngày")
ax.set_title("Bản đồ nhiệt: Mật độ khách theo tầng / căn hộ\n★=AirBnb ▲=Nghi ngờ", fontsize=12)
plt.tight_layout()
plt.savefig("data/charts/08_floor_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[CHART] 08_floor_heatmap.png")

print("\n✅ Hoàn tất! Các file đã tạo:")
print("   data/dataset_full.csv")
print("   data/dataset_summary.csv")
print("   data/daily_trend.csv")
print("   data/charts/ (8 charts PNG)")
