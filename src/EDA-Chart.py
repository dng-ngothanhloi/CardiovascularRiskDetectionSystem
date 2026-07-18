import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.utils import get_project_root
except ImportError:
    from utils import get_project_root

# Get project root and construct data path
project_root = get_project_root()
data_path = project_root / "data" / "cardio_data.csv"

# Load dữ liệu - auto-detect delimiter
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

# Try to detect delimiter (semicolon or comma)
with open(data_path, 'r') as f:
    first_line = f.readline()
    if ';' in first_line:
        delimiter = ';'
    else:
        delimiter = ','

df = pd.read_csv(data_path, sep=delimiter)
df.columns = [c.strip().lower() for c in df.columns]

# Tạo biến BMI, Pulse Pressure và tuổi (năm)
df["bmi"] = df["weight"] / ((df["height"]/100)**2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
# age trong CSV thường tính bằng ngày
if df["age"].median() > 200:
    df["age_new"] = (df["age"] / 365.25).astype(int)
else:
    df["age_new"] = df["age"].astype(int)

# BMI_State: Normal (18.5–24.9) vs Abnormal (thiếu cân / thừa cân / béo phì)
df["BMI_State"] = np.where(
    (df["bmi"] >= 18.5) & (df["bmi"] < 25.0),
    "Normal",
    "Abnormal",
)

# Helper: Vẽ xu hướng tỷ lệ mắc bệnh tim theo biến
def cardio_rate_by_feature(df, feature, bins, xlabel, filename):
    df[f"{feature}_bin"] = pd.cut(df[feature], bins=bins)
    rate = df.groupby(f"{feature}_bin", observed=True)["cardio"].mean().reset_index()
    
    plt.figure(figsize=(6,4))
    sns.barplot(x=f"{feature}_bin", y="cardio", data=df, errorbar=None, color="#AED6F1", alpha=0.6)
    sns.lineplot(x=range(len(rate)), y=rate["cardio"], marker="o", color="#E67E22", linewidth=2.0)
    plt.title(f"Tỷ lệ mắc bệnh tim mạch theo {xlabel}")
    plt.ylabel("Tỷ lệ bệnh tim (%)")
    plt.xlabel(xlabel)
    plt.xticks(range(len(rate)), [f"{i.left:.0f}-{i.right:.0f}" for i in rate[f"{feature}_bin"]], rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# Tạo thư mục output nếu chưa có
output_dir = project_root / "artifacts" / "eda-added"
output_dir.mkdir(parents=True, exist_ok=True)

# Tạo 4 biểu đồ riêng
cardio_rate_by_feature(df, "ap_hi", range(80, 201, 10), "Huyết áp tâm thu (ap_hi)", str(output_dir / "cardio_rate_ap_hi.png"))
cardio_rate_by_feature(df, "ap_lo", range(50, 121, 10), "Huyết áp tâm trương (ap_lo)", str(output_dir / "cardio_rate_ap_lo.png"))
cardio_rate_by_feature(df, "bmi", [15, 18.5, 25, 30, 35, 40, 60], "Chỉ số khối cơ thể (BMI)", str(output_dir / "cardio_rate_bmi.png"))
cardio_rate_by_feature(df, "pulse_pressure", range(20, 101, 10), "Chênh lệch huyết áp (Pulse Pressure)", str(output_dir / "cardio_rate_pulse_pressure.png"))

# --- Tổng hợp 4 biểu đồ trong 1 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = [
    ("ap_hi", "Huyết áp tâm thu (ap_hi)", range(80, 201, 10)),
    ("ap_lo", "Huyết áp tâm trương (ap_lo)", range(50, 121, 10)),
    ("bmi", "Chỉ số khối cơ thể (BMI)", [15, 18.5, 25, 30, 35, 40, 60]),
    ("pulse_pressure", "Chênh lệch huyết áp (Pulse Pressure)", range(20, 101, 10))
]

for ax, (feature, xlabel, bins) in zip(axes.flatten(), features):
    df[f"{feature}_bin"] = pd.cut(df[feature], bins=bins)
    rate = df.groupby(f"{feature}_bin", observed=True)["cardio"].mean().reset_index()
    sns.barplot(x=f"{feature}_bin", y="cardio", data=df, errorbar=None, color="#AED6F1", alpha=0.6, ax=ax)
    sns.lineplot(x=range(len(rate)), y=rate["cardio"], marker="o", color="#E67E22", linewidth=2.0, ax=ax)
    ax.set_title(xlabel)
    ax.set_ylabel("Tỷ lệ mắc bệnh tim (%)")
    ax.set_xlabel(xlabel)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{i.left:.0f}-{i.right:.0f}" for i in rate[f"{feature}_bin"]], rotation=45)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(output_dir / "EDA_TrendOverview.png"), dpi=200)
# plt.show()  # Comment out to avoid blocking when running as script
plt.close()

# --- Tỷ lệ / phân bố CVD theo độ tuổi (countplot + đường xu hướng cardio=1)
def plot_cardio_by_age(df: pd.DataFrame, out_path: str):
    """
    Grouped countplot theo age_new, hue=cardio, kèm đường làm mượt số lượng cardio=1.
    """
    plot_df = df[df["cardio"].isin([0, 1])].copy()
    order = sorted(plot_df["age_new"].unique())

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    sns.countplot(
        data=plot_df,
        x="age_new",
        hue="cardio",
        order=order,
        palette={0: "#2E8B57", 1: "#E67E22"},
        ax=ax,
    )

    # Đường xu hướng theo số lượng cardio=1 tại mỗi tuổi
    counts_1 = (
        plot_df.loc[plot_df["cardio"] == 1, "age_new"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(float)
        .values
    )
    x_pos = np.arange(len(order), dtype=float)
    if len(order) >= 4 and counts_1.sum() > 0:
        deg = min(5, len(order) - 1)
        coef = np.polyfit(x_pos, counts_1, deg=deg)
        x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)
        y_smooth = np.polyval(coef, x_smooth)
        y_smooth = np.clip(y_smooth, 0, None)
        ax.plot(x_smooth, y_smooth, color="#8B0000", linewidth=2.2, label="Xu hướng cardio=1")

    ax.set_title("Tỷ lệ mắc bệnh tim mạch theo độ tuổi")
    ax.set_xlabel("age_new")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="cardio", loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.95,
        "Tỉ lệ CVD tăng dần từ tuổi trung niên",
        transform=ax.transAxes,
        color="red",
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Tránh chồng chữ khi nhiều tuổi
    if len(order) > 20:
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 2 != 0:
                label.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


plot_cardio_by_age(df, str(output_dir / "cardio_by_age.png"))


def plot_cardio_by_bmi_state(df: pd.DataFrame, out_path: str):
    """
    Countplot BMI_State (Normal/Abnormal) x cardio, kèm mũi tên xu hướng và chú thích nguy cơ.
    """
    plot_df = df[df["cardio"].isin([0, 1])].copy()
    order = ["Normal", "Abnormal"]
    plot_df = plot_df[plot_df["BMI_State"].isin(order)]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    sns.countplot(
        data=plot_df,
        x="BMI_State",
        hue="cardio",
        order=order,
        hue_order=[0, 1],
        palette={0: "#2E8B57", 1: "#E67E22"},
        ax=ax,
    )

    # Lấy tâm đỉnh từng cột từ patches (thứ tự: Normal/0, Normal/1, Abnormal/0, Abnormal/1)
    bar_tops = []
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        if width <= 0 or height <= 0:
            continue
        x_center = patch.get_x() + width / 2.0
        bar_tops.append((x_center, height))
    bar_tops = sorted(bar_tops, key=lambda t: t[0])

    if len(bar_tops) >= 4:
        (x_n0, y_n0), (x_n1, y_n1), (x_a0, y_a0), (x_a1, y_a1) = bar_tops[:4]
        ax.annotate(
            "",
            xy=(x_a0, y_a0),
            xytext=(x_n0, y_n0),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.0),
        )
        ax.annotate(
            "",
            xy=(x_a1, y_a1),
            xytext=(x_n1, y_n1),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.5),
        )

    ax.set_title("Nguy cơ CVD theo trạng thái BMI")
    ax.set_xlabel("BMI_State")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="cardio", loc="upper left", frameon=False)
    ax.text(
        0.98,
        0.95,
        "Người có BMI bất thường thì nguy cơ CVD tăng",
        transform=ax.transAxes,
        color="red",
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="right",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


plot_cardio_by_bmi_state(df, str(output_dir / "cardio_by_bmi_state.png"))

"""
Vẽ 2 biểu đồ EDA cho biến ordinal (1–3) với đường xu hướng:
  - ordinal_trend_cholesterol.png
  - ordinal_trend_gluc.png

Y-axis: tỷ lệ mắc CVD (cardio=1) theo từng mức (1,2,3).
Dùng thuần matplotlib; không phụ thuộc seaborn.
"""

# -----------------------------
# Load & chuẩn hóa cột
# -----------------------------

required = {"cardio", "cholesterol", "gluc"}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Thiếu cột bắt buộc: {missing}")

# Giữ lại các giá trị hợp lệ 1,2,3 cho 2 biến ordinal
df = df[df["cholesterol"].isin([1, 2, 3]) & df["gluc"].isin([1, 2, 3])]

# -----------------------------
# Hàm vẽ chung cho biến ordinal
# -----------------------------
def plot_ordinal_trend(
    df: pd.DataFrame,
    col: str,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str = "Tỷ lệ mắc bệnh tim (cardio=1)",
    show_percent: bool = True,
):
    """
    Vẽ cột + đường xu hướng cho biến ordinal với các mức 1,2,3.
    - Cột: tỷ lệ trung bình cardio=1 theo từng mức.
    - Đường xu hướng: hồi quy tuyến tính (polyfit bậc 1) qua các điểm (1,2,3).
    """
    # Tính tỷ lệ cardio=1 theo từng mức
    stats = df.groupby(col)["cardio"].mean().reindex([1, 2, 3])
    x_levels = np.array([1, 2, 3], dtype=float)
    y_vals = stats.values.astype(float)

    # Chuyển sang % nếu muốn
    if show_percent:
        y_plot = y_vals * 100.0
        y_label = ylabel + " (%)"
    else:
        y_plot = y_vals
        y_label = ylabel

    # Hồi quy tuyến tính cho đường trend (bậc 1)
    # y ≈ a*x + b
    a, b = np.polyfit(x_levels, y_plot, deg=1)
    y_fit = a * x_levels + b

    # Vẽ
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    # Cột (bar)
    ax.bar(x_levels, y_plot, width=0.6, alpha=0.6, label="Tỷ lệ theo mức (1–3)")

    # Đường trend (line)
    ax.plot(x_levels, y_plot, marker="o", linewidth=2.0, label="Giá trị quan sát")
    ax.plot(x_levels, y_fit, linestyle="--", linewidth=2.0, label="Đường xu hướng (OLS)")

    # Trang trí
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1", "2", "3"])
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="best", frameon=False)

    # Ghi chú hệ số xu hướng (tuỳ chọn)
    # Dấu dương của 'a' → xu hướng tăng; âm → giảm
    ax.text(
        0.04, 0.92,
        f"Trend slope a = {a:.2f} (%/mức)",
        transform=ax.transAxes,
        fontsize=10,
        ha="left", va="center",
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# -----------------------------
# Vẽ 2 biểu đồ theo yêu cầu
# -----------------------------
plot_ordinal_trend(
    df=df,
    col="cholesterol",
    out_path=str(output_dir / "ordinal_trend_cholesterol.png"),
    title="Xu hướng tỷ lệ CVD theo mức Cholesterol (1–3)",
    xlabel="Mức cholesterol: 1=Bình thường, 2=Tăng nhẹ, 3=Tăng cao",
)

plot_ordinal_trend(
    df=df,
    col="gluc",
    out_path=str(output_dir / "ordinal_trend_gluc.png"),
    title="Xu hướng tỷ lệ CVD theo mức Glucose máu (1–3)",
    xlabel="Mức glucose: 1=Bình thường, 2=Tăng nhẹ, 3=Tăng cao",
)
