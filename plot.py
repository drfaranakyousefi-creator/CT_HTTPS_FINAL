import math
import os
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display

# ── Visual constants ──────────────────────────────────────────────
_COLORS = {
    "train": "#2563EB",
    "test":  "#DC2626",
}

_METRIC_META = {
    "MSE":  {
        "label":   "MSE — Mean Squared Error",
        "y_label": "MSE",
        "better":  "Lower is better",
        "unit":    "",
        "filename": "metric_MSE.png",
        "fa_desc": "این نمودار MSE است — میانگین مربع خطا، هرچه کمتر بهتر",
    },
    "RMSE": {
        "label":   "RMSE — Root Mean Squared Error",
        "y_label": "RMSE",
        "better":  "Lower is better",
        "unit":    "",
        "filename": "metric_RMSE.png",
        "fa_desc": "این نمودار RMSE است — ریشه میانگین مربع خطا، هرچه کمتر بهتر",
    },
    "MAE":  {
        "label":   "MAE — Mean Absolute Error",
        "y_label": "MAE",
        "better":  "Lower is better",
        "unit":    "",
        "filename": "metric_MAE.png",
        "fa_desc": "این نمودار MAE است — میانگین قدرمطلق خطا، به outlier مقاوم‌تر",
    },
    "R2":   {
        "label":   "R² — Coefficient of Determination",
        "y_label": "R²",
        "better":  "Closer to 1 is better",
        "unit":    "",
        "filename": "metric_R2.png",
        "fa_desc": "این نمودار R² است — ضریب تعیین، هرچه به ۱ نزدیک‌تر بهتر",
    },
    "MAPE": {
        "label":   "MAPE — Mean Absolute Percentage Error",
        "y_label": "MAPE (%)",
        "better":  "Lower is better",
        "unit":    "%",
        "filename": "metric_MAPE.png",
        "fa_desc": "این نمودار MAPE است — میانگین درصد خطای مطلق، هرچه کمتر بهتر",
    },
}


# ── Public entry point ────────────────────────────────────────────
def plot_history(history: dict, save_dir: str = "."):
    """
    Plot and save training history as separate image files.
    Also displays each figure inline when running in a Jupyter Notebook.

    Parameters
    ----------
    history  : dict returned by CT_HTTPS.fit()
    save_dir : folder where .png files will be saved (default: current dir)

    Saved files
    -----------
    metric_MSE.png, metric_RMSE.png, metric_MAE.png,
    metric_R2.png,  metric_MAPE.png, summary_table.png
    """
    os.makedirs(save_dir, exist_ok=True)

    if "train_RMSE" not in history:
        _plot_simple(history, save_dir)
        return

    epochs = list(range(1, len(history["loss_train"]) + 1))
    saved  = []

    for key in _METRIC_META:
        path = _plot_single_metric(history, key, epochs, save_dir)
        saved.append(path)

    path = _plot_summary_table(history, save_dir)
    saved.append(path)

    print("\n── Saved figures ─────────────────────────────")
    for p in saved:
        print(f"   {p}")
    print("──────────────────────────────────────────────\n")


# ── Legacy ────────────────────────────────────────────────────────
def _plot_simple(history, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#F8FAFC")

    epochs = list(range(1, len(history["loss_train"]) + 1))
    ax.plot(epochs, history["loss_train"],
            color=_COLORS["train"], linewidth=2.2,
            marker="o", markersize=4, label="Train Loss")
    ax.plot(epochs, history["loss_test"],
            color=_COLORS["test"], linewidth=2.2,
            marker="s", markersize=4, linestyle="--", label="Test Loss")

    ax.set_title("Training vs Testing — MSE Loss",
                 fontsize=14, fontweight="bold", color="#1E293B", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_facecolor("#FFFFFF")
    _style_spines(ax)

    plt.tight_layout()
    path = os.path.join(save_dir, "metric_MSE.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")

    print("این نمودار MSE است — میانگین مربع خطا برای train و test")
    ipy_display(fig)
    plt.close(fig)
    return path


# ── One figure per metric ─────────────────────────────────────────
def _plot_single_metric(history, key, epochs, save_dir):
    meta    = _METRIC_META[key]
    tr_vals = history[f"train_{key}"]
    te_vals = history[f"test_{key}"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#F8FAFC")

    ax.plot(epochs, tr_vals,
            color=_COLORS["train"], linewidth=2.2,
            marker="o", markersize=5, label="Train")
    ax.plot(epochs, te_vals,
            color=_COLORS["test"], linewidth=2.2,
            marker="s", markersize=5, linestyle="--", label="Test")

    _annotate_last(ax, epochs, tr_vals, _COLORS["train"], above=True,  unit=meta["unit"])
    _annotate_last(ax, epochs, te_vals, _COLORS["test"],  above=False, unit=meta["unit"])

    ax.set_title(
        f"{meta['label']}\n{meta['better']}",
        fontsize=13, fontweight="bold", color="#1E293B", pad=10
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(meta["y_label"], fontsize=11)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_facecolor("#FFFFFF")
    _style_spines(ax)

    plt.tight_layout()
    path = os.path.join(save_dir, meta["filename"])
    fig.savefig(path, dpi=150, bbox_inches="tight")

    # نمایش inline توی notebook با توضیح فارسی
    print(meta["fa_desc"])
    ipy_display(fig)
    plt.close(fig)
    return path


# ── Summary table ─────────────────────────────────────────────────
def _plot_summary_table(history, save_dir):
    metrics    = list(_METRIC_META.keys())
    col_labels = ["Metric", "Train (last)", "Test (last)", "Better when"]
    rows = []
    for key in metrics:
        meta = _METRIC_META[key]
        tr_v = history[f"train_{key}"][-1]
        te_v = history[f"test_{key}"][-1]
        unit = meta["unit"]

        def _fmt(v, u=unit):
            return "N/A" if math.isnan(v) else f"{v:.4f}{u}"

        rows.append([meta["y_label"], _fmt(tr_v), _fmt(te_v), meta["better"]])

    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#F8FAFC")
    ax.axis("off")

    fig.suptitle("CT-HTTPS — Final Epoch Summary",
                 fontsize=14, fontweight="bold", color="#1E293B", y=0.97)

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.15, 1.7)

    for col_idx in range(len(col_labels)):
        cell = tbl[(0, col_idx)]
        cell.set_facecolor("#1E40AF")
        cell.set_text_props(color="white", fontweight="bold")

    row_colors = ["#EFF6FF", "#DBEAFE"]
    for row_idx in range(1, len(rows) + 1):
        for col_idx in range(len(col_labels)):
            tbl[(row_idx, col_idx)].set_facecolor(
                row_colors[(row_idx - 1) % 2]
            )

    plt.tight_layout()
    path = os.path.join(save_dir, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")

    # نمایش inline توی notebook با توضیح فارسی
    print("این جدول خلاصه نهایی است — مقادیر آخرین epoch برای همه معیارها")
    ipy_display(fig)
    plt.close(fig)
    return path


# ── Helpers ───────────────────────────────────────────────────────
def _annotate_last(ax, epochs, vals, color, above, unit):
    last_val = vals[-1]
    if math.isnan(last_val):
        return
    offset = (4, 6) if above else (4, -12)
    ax.annotate(
        f"{last_val:.3f}{unit}",
        xy=(epochs[-1], last_val),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.5, color=color, fontweight="bold",
    )


def _style_spines(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
