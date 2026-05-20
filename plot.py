import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# تنظیمات بصری ثابت
_COLORS = {
    "train": "#2563EB",   # آبی
    "test":  "#DC2626",   # قرمز
}
_METRIC_META = {
    "MSE":  {"label": "MSE",        "better": "↓ پایین‌تر بهتر",  "unit": ""},
    "RMSE": {"label": "RMSE",       "better": "↓ پایین‌تر بهتر",  "unit": ""},
    "MAE":  {"label": "MAE",        "better": "↓ پایین‌تر بهتر",  "unit": ""},
    "R2":   {"label": "R²",         "better": "↑ به ۱ نزدیک‌تر", "unit": ""},
    "MAPE": {"label": "MAPE (%)",   "better": "↓ پایین‌تر بهتر",  "unit": "%"},
}


def plot_history(history: dict):
    """
    رسم کامل تاریخچه آموزش.

    اگر history فقط شامل loss_train / loss_test باشد (حالت قدیمی)،
    فقط همان نمودار MSE رسم می‌شود.
    اگر کلیدهای کامل وجود داشته باشند، داشبورد ۶ نمودار نمایش می‌دهد.
    """
    has_full = "train_RMSE" in history

    if not has_full:
        _plot_simple(history)
    else:
        _plot_dashboard(history)


# ──────────────────────────────────────────────────────────────────
#  حالت قدیمی (فقط loss)
# ──────────────────────────────────────────────────────────────────
def _plot_simple(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss_train"], label="Train Loss",
            color=_COLORS["train"], linewidth=2)
    ax.plot(history["loss_test"],  label="Test Loss",
            color=_COLORS["test"],  linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training vs Testing Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────
#  داشبورد کامل (۵ معیار + خلاصه)
# ──────────────────────────────────────────────────────────────────
def _plot_dashboard(history):
    metrics   = list(_METRIC_META.keys())          # MSE, RMSE, MAE, R2, MAPE
    n_metrics = len(metrics)                        # 5
    epochs    = list(range(1, len(history["loss_train"]) + 1))

    # چیدمان: ردیف اول ۳ نمودار، ردیف دوم ۲ نمودار + جدول خلاصه
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#F8FAFC")

    # عنوان اصلی
    fig.suptitle(
        "CT-HTTPS — نتایج ارزیابی مدل",
        fontsize=17, fontweight="bold", color="#1E293B", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        left=0.06, right=0.97,
        top=0.91, bottom=0.08,
    )

    axes_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1),
    ]

    for idx, key in enumerate(metrics):
        row, col = axes_positions[idx]
        ax = fig.add_subplot(gs[row, col])
        meta = _METRIC_META[key]

        tr_vals = history[f"train_{key}"]
        te_vals = history[f"test_{key}"]

        ax.plot(epochs, tr_vals,
                color=_COLORS["train"], linewidth=2.2,
                marker="o", markersize=4, label="Train")
        ax.plot(epochs, te_vals,
                color=_COLORS["test"],  linewidth=2.2,
                marker="s", markersize=4, label="Test", linestyle="--")

        # آخرین مقدار روی نمودار
        ax.annotate(f"{tr_vals[-1]:.3f}",
                    xy=(epochs[-1], tr_vals[-1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=_COLORS["train"])
        ax.annotate(f"{te_vals[-1]:.3f}",
                    xy=(epochs[-1], te_vals[-1]),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=8, color=_COLORS["test"])

        ax.set_title(
            f"{meta['label']}   {meta['better']}",
            fontsize=10.5, fontweight="bold", color="#334155", pad=6
        )
        ax.set_xlabel("Epoch", fontsize=9, color="#475569")
        ax.set_ylabel(meta["label"] + (f" ({meta['unit']})" if meta["unit"] else ""),
                      fontsize=9, color="#475569")
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_facecolor("#FFFFFF")
        for spine in ax.spines.values():
            spine.set_edgecolor("#CBD5E1")

    # ── جدول خلاصه آخرین epoch ──────────────────────────────────
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis("off")

    col_labels = ["Metric", "Train", "Test", "Better"]
    table_data = []
    for key in metrics:
        meta   = _METRIC_META[key]
        tr_v   = history[f"train_{key}"][-1]
        te_v   = history[f"test_{key}"][-1]
        tr_str = f"{tr_v:.4f}" + (meta["unit"] if meta["unit"] else "")
        te_str = f"{te_v:.4f}" + (meta["unit"] if meta["unit"] else "")
        table_data.append([meta["label"], tr_str, te_str, meta["better"]])

    tbl = ax_table.table(
        cellText=col_labels,
        cellLoc="center",
        loc="upper center",
    )

    # رسم دستی جدول با زیبایی بیشتر
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.55)

    # رنگ‌بندی هدر
    for col_idx in range(len(col_labels)):
        tbl[(0, col_idx)].set_facecolor("#1E40AF")
        tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")

    # رنگ زیبا برای ردیف‌های داده
    row_colors = ["#EFF6FF", "#DBEAFE"]
    for row_idx in range(1, len(metrics) + 1):
        for col_idx in range(len(col_labels)):
            tbl[(row_idx, col_idx)].set_facecolor(
                row_colors[(row_idx - 1) % 2]
            )

    ax_table.set_title(
        f"خلاصه — Epoch آخر",
        fontsize=10.5, fontweight="bold", color="#334155", pad=8
    )

    plt.show()
