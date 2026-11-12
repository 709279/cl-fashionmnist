"""持续学习报告的图表。"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cl_data import TASK_CLASSES, CLASS_CN  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
RUNS = os.path.join(ROOT, "results_cl", "runs")

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

DISPLAY = {"naive": "顺序微调", "replay": "经验回放", "ewc": "EWC", "lwf": "LwF",
           "joint": "联合训练（上限）"}
SHORT = {"naive": "Naive", "replay": "Replay", "ewc": "EWC", "lwf": "LwF"}
COLORS = {"naive": "#C44E52", "replay": "#55A868", "ewc": "#8172B3", "lwf": "#DD8452",
          "joint": "#666666"}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".png")
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print("saved", name)


def load(tag):
    with open(os.path.join(RUNS, f"cl_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


def seeds(method):
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, f"cl_{method}_s*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _matrix(r):
    return np.array([[np.nan if v is None else v for v in row] for row in r["acc_matrix"]])


# ---------------- 图 1：任务序列与样本 ----------------
def fig_tasks():
    from data import load_base

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.25),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    for t, cls in enumerate(TASK_CLASSES):
        y = 4 - t
        ax.barh(y, 1.0, height=0.62, left=0, color=plt.cm.Blues(0.30 + 0.11 * t),
                edgecolor="white", linewidth=0.8)
        ax.text(0.03, y, f"任务 {t+1}", va="center", ha="left", fontsize=8,
                color="white", fontweight="bold")
        ax.text(0.30, y, "＋".join(CLASS_CN[c] for c in cls), va="center", ha="left",
                fontsize=7.6, color="white")
        ax.text(0.97, y, "12k", va="center", ha="right", fontsize=6.8, color="white")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, 4.7)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.annotate("", xy=(1.0, -0.55), xytext=(0.0, -0.55),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#444"))
    ax.text(0.5, -0.85, "训练顺序（同一模型依次学习，学新任务时看不到旧数据）",
            ha="center", fontsize=6.8, color="#444")
    ax.set_title("(a) Split-Fashion-MNIST 任务序列")

    ax = axes[1]
    imgs, labels, _, _ = load_base("fashion", True)
    labs = labels.numpy()
    cell = 32
    canvas = np.ones((cell * 5 + 4, cell * 2 + 4), dtype=np.float32)
    for t, cls in enumerate(TASK_CLASSES):
        for j, c in enumerate(cls):
            idx = np.where(labs == c)[0]
            img = imgs[idx[3 * t + 5], 0].float().numpy() / 255.0
            y0 = t * cell + 2
            x0 = j * cell + 2
            canvas[y0:y0 + 28, x0:x0 + 28] = img
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([cell * 0.5 + 1, cell * 1.5 + 1])
    ax.set_xticklabels(["类 1", "类 2"], fontsize=7.5)
    ax.set_yticks([cell * t + cell / 2 for t in range(5)])
    ax.set_yticklabels([f"任务{t+1}" for t in range(5)], fontsize=7.5)
    ax.grid(False)
    ax.set_title("(b) 每个任务的类别样本（类名见表 1）")
    fig.tight_layout()
    save(fig, "cl_fig1_tasks")


# ---------------- 图 2：遗忘曲线 ----------------
def _curve(r):
    R = _matrix(r)
    return [np.nanmean(R[t, : t + 1]) for t in range(R.shape[0])]


def fig_forgetting():
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    xs = np.arange(1, 6)
    for m in ("naive", "lwf", "ewc", "replay"):
        rs = seeds(m)
        if not rs:
            continue
        C = np.array([_curve(r) for r in rs]) * 100
        ax.plot(xs, C.mean(0), marker="o", markersize=3.2, linewidth=1.3,
                color=COLORS[m], label=DISPLAY[m])
        ax.fill_between(xs, C.min(0), C.max(0), color=COLORS[m], alpha=0.15, linewidth=0)
    jr = seeds("joint")
    if jr:
        j = np.mean([r["metrics"]["acc"] for r in jr]) * 100
        ax.axhline(j, color=COLORS["joint"], linestyle="--", linewidth=1.0)
        ax.text(3.05, j + 2.5, f"联合训练上限 {j:.1f}%", fontsize=6.4, color=COLORS["joint"])
    ax.set_xticks(xs)
    ax.set_xlabel("已完成的任务数")
    ax.set_ylabel("已学任务平均准确率 (%)")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=6.4, loc="lower left", handlelength=1.3,
              bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.1)
    fig.tight_layout()
    save(fig, "cl_fig2_forgetting")


# ---------------- 图 3：准确率矩阵 ----------------
def fig_matrix():
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
    for k, (ax, m) in enumerate(zip(axes, ("naive", "replay"))):
        rs = seeds(m)
        R = np.nanmean([_matrix(r) for r in rs], axis=0) * 100
        im = ax.imshow(R, cmap="YlGnBu", vmin=0, vmax=100)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if np.isnan(R[i, j]):
                    continue
                ax.text(j, i, f"{R[i,j]:.0f}", ha="center", va="center", fontsize=6.4,
                        color="white" if R[i, j] > 55 else "#222")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"T{i+1}" for i in range(5)], fontsize=7.2)
        ax.set_yticks(range(5))
        if k == 0:
            ax.set_yticklabels([f"学完 T{i+1}" for i in range(5)], fontsize=7.2)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("被评测的任务")
        ax.set_title(f"({'ab'[k]}) {DISPLAY[m]}")
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.025)
    cb.set_label("准确率 (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "cl_fig3_matrix")


# ---------------- 图 4：主结果 ----------------
def fig_main():
    methods = ["naive", "lwf", "ewc", "replay"]
    accs, bwts, acc_err, bwt_err = [], [], [], []
    for m in methods:
        rs = seeds(m)
        a = np.array([r["metrics"]["acc"] for r in rs]) * 100
        b = np.array([r["metrics"]["bwt"] for r in rs]) * 100
        accs.append(a.mean())
        acc_err.append(a.std())
        bwts.append(b.mean())
        bwt_err.append(b.std())

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.35))
    xs = np.arange(len(methods))

    ax = axes[0]
    bars = ax.bar(xs, accs, yerr=acc_err, capsize=2.2, width=0.6,
                  color=[COLORS[m] for m in methods], edgecolor="white",
                  error_kw=dict(lw=0.7, capthick=0.7))
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}", ha="center", fontsize=6.4)
    jr = seeds("joint")
    if jr:
        jv = np.mean([r["metrics"]["acc"] for r in jr]) * 100
        ax.axhline(jv, color=COLORS["joint"], linestyle="--", linewidth=1.0)
        ax.text(0.62, jv - 4.5, f"联合训练上限 {jv:.1f}", fontsize=6.3,
                color=COLORS["joint"], va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in methods], fontsize=7)
    ax.set_ylabel("准确率 (%)")
    ax.set_ylim(0, 112)
    ax.set_title("(a) 最终平均准确率")

    ax = axes[1]
    bars = ax.bar(xs, bwts, yerr=bwt_err, capsize=2.2, width=0.6,
                  color=[COLORS[m] for m in methods], edgecolor="white",
                  error_kw=dict(lw=0.7, capthick=0.7))
    for b, v in zip(bars, bwts):
        ax.text(b.get_x() + b.get_width() / 2, v - 9, f"{v:.0f}", ha="center",
                fontsize=6.4, color="white" if v < 0 else "#222")
    ax.axhline(0, color="#888", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in methods], fontsize=7)
    ax.set_ylabel("BWT (%)")
    ax.set_ylim(min(bwts) - 26, max(12, max(bwts) + 18))
    ax.set_title("(b) 后向迁移 BWT")

    fig.tight_layout()
    save(fig, "cl_fig4_main")


# ---------------- 图 5：崩塌与消融 ----------------
def fig_collapse():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2))

    ax = axes[0]
    w = 0.38
    for k, m in enumerate(("naive", "replay")):
        rs = seeds(m)
        h = np.mean([np.array(r["pred_hist_task1"], dtype=float) for r in rs], axis=0)
        h = h / h.sum() * 100
        ax.bar(np.arange(10) + (k - 0.5) * w, h, width=w, color=COLORS[m],
               label=DISPLAY[m], edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(c) for c in range(10)], fontsize=6.2)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("占比 (%)")
    ax.set_title("(a) 学完 5 个任务后对任务 1 的预测")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.set_ylim(0, 118)

    ax = axes[1]
    sizes = [10, 40, 100]
    tags = ["replay_pc10_s0", "replay_s0", "replay_pc100_s0"]
    acc = []
    for tg in tags:
        try:
            acc.append(load(tg)["metrics"]["acc"] * 100)
        except FileNotFoundError:
            acc.append(np.nan)
    ax.plot(sizes, acc, marker="o", markersize=4, linewidth=1.4, color=COLORS["replay"])
    for x, y in zip(sizes, acc):
        if not np.isnan(y):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6.4)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("每类缓存样本数")
    ax.set_ylabel("最终准确率 (%)")
    ax.set_title("(b) 回放缓存大小的影响")
    ax.set_ylim(0, 100)

    ax = axes[2]
    lams = [100, 1000, 10000]
    tags = ["ewc_lam100_s0", "ewc_s0", "ewc_lam10000_s0"]
    acc = []
    for tg in tags:
        try:
            acc.append(load(tg)["metrics"]["acc"] * 100)
        except FileNotFoundError:
            acc.append(np.nan)
    ax.plot(lams, acc, marker="s", markersize=4, linewidth=1.4, color=COLORS["ewc"])
    for x, y in zip(lams, acc):
        if not np.isnan(y):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6.4)
    ax.set_xscale("log")
    ax.set_xticks(lams)
    ax.set_xticklabels([str(l) for l in lams])
    ax.set_xlabel("EWC 惩罚系数 λ")
    ax.set_ylabel("最终准确率 (%)")
    ax.set_title("(c) EWC 惩罚强度的影响")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    save(fig, "cl_fig5_collapse")


if __name__ == "__main__":
    fig_tasks()
    fig_forgetting()
    fig_matrix()
    fig_main()
    fig_collapse()
