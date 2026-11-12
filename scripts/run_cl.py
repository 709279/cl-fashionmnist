"""持续学习实验调度：Split-Fashion-MNIST 上的四种抗遗忘策略。

用法：
    python scripts/run_cl.py main 0 1      # 分片运行（第 0 片 / 共 1 片）
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch

from common import set_seed, save_json, ensure_dirs
from models import build_model
from cl_data import build_tasks, eval_matrix_row
from cl_methods import (ReplayBuffer, fisher_diagonal, snapshot, clone_for_teacher,
                        train_task)
from cl_data import predict_all

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_cl", "runs")

BASE = dict(dataset="fashion", num_classes=10, width=32, dropout=0.3,
            batch_size=512, epochs=3, lr=1e-3, wd=5e-4, per_task=0)


def run_one(cfg, verbose=True):
    cfg = {**BASE, **cfg}
    ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    set_seed(cfg["seed"])
    dev = torch.device("cpu")

    bundle = build_tasks(cfg, cfg["seed"])
    n_tasks = bundle["meta"]["n_tasks"]
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    mean = torch.tensor([0.2860], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor([0.3530], dtype=torch.float32).view(1, -1, 1, 1)

    state = dict(mean=mean, std=std)
    if cfg["method"] == "replay":
        state["replay_buffer"] = ReplayBuffer(cfg.get("per_class", 40), seed=cfg["seed"])
    if cfg["method"] == "ewc":
        state["ewc_terms"] = []
        state["ewc_lambda"] = cfg.get("ewc_lambda", 1000.0)
    if cfg["method"] == "lwf":
        state["lwf_alpha"] = cfg.get("lwf_alpha", 1.0)
        state["lwf_T"] = cfg.get("lwf_T", 2.0)

    R, losses = [], []
    t0 = time.time()

    if cfg["method"] == "joint":
        for g in opt.param_groups:
            g["lr"] = cfg["lr"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.02)
        for ep in range(cfg["epochs"]):
            l = train_task(model, bundle["joint"], "naive", opt, 1, dev, state,
                           sched=sched)
            losses.append(l)          # 与逐任务分支保持相同结构：[[每个 epoch 的损失], ...]
        row = eval_matrix_row(model, bundle["test"], dev, n_tasks - 1)
        R = [row]
        for _ in range(n_tasks - 1):
            R.insert(0, [None] * n_tasks)
    else:
        for t in range(n_tasks):
            if cfg["method"] == "lwf" and t > 0:
                state["old_model"] = clone_for_teacher(model)

            for g in opt.param_groups:      # 每个任务都从基准学习率重新退火
                g["lr"] = cfg["lr"]
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.02)
            l = train_task(model, bundle["train"][t], cfg["method"], opt,
                           cfg["epochs"], dev, state, sched=sched)
            losses.append(l)
            row = eval_matrix_row(model, bundle["test"], dev, t)
            R.append(row)

            if verbose:
                seen = " ".join(f"{v*100:5.1f}" for v in row)
                print(f"    task {t+1}/{n_tasks} loss {np.mean(l):.4f} | acc {seen}",
                      flush=True)

            # 方法状态更新
            if cfg["method"] == "replay":
                imgs, labs = bundle["raw"][t]
                state["replay_buffer"].add_task(imgs, labs.numpy())
            if cfg["method"] == "ewc":
                f = fisher_diagonal(model, bundle["train"][t], dev,
                                    n_batches=cfg.get("fisher_batches", 4))
                state["ewc_terms"].append((f, snapshot(model)))

    # ---- 指标 ----
    final = R[-1]
    acc = float(np.mean(final))
    bwt_terms, forget_terms = [], []
    if cfg["method"] != "joint":      # 联合训练没有「逐任务学习」过程
        for i in range(n_tasks - 1):
            col = [R[t][i] for t in range(i, n_tasks)]
            bwt_terms.append(final[i] - R[i][i])
            forget_terms.append(max(col) - final[i])
    bwt = float(np.mean(bwt_terms)) if bwt_terms else 0.0
    forgetting = float(np.mean(forget_terms)) if forget_terms else 0.0

    # 分布崩塌分析：学完所有任务后，在第 1 个任务的测试集上预测成了什么
    logits0, y0 = predict_all(model, bundle["test"][0], dev)
    hist0 = np.bincount(logits0.argmax(1), minlength=10).tolist()

    out = {
        "cfg": {k: v for k, v in cfg.items()},
        "meta": bundle["meta"],
        "acc_matrix": [[None if v is None else round(v, 6) for v in row] for row in R],
        "train_loss": [[round(x, 5) for x in l] for l in losses],
        "metrics": {"acc": acc, "bwt": bwt, "forgetting": forgetting,
                    "per_task_final": [round(v, 6) for v in final]},
        "pred_hist_task1": hist0,
        "seconds": round(time.time() - t0, 1),
    }
    tag = cfg.get("tag") or f"{cfg['method']}_s{cfg['seed']}"
    save_json(out, os.path.join(OUT, f"cl_{tag}.json"))
    print(f"  -> acc {acc*100:.2f} bwt {bwt*100:+.2f} "
          f"forget {forgetting*100:.2f} ({out['seconds']}s)", flush=True)
    return out


def jobs_main():
    out = []
    for m in ("naive", "replay", "ewc", "lwf"):
        for s in (0, 1):
            out.append(dict(method=m, seed=s))
    out.append(dict(method="joint", seed=0))
    return out


def jobs_ablation():
    out = []
    for pc in (10, 100):
        out.append(dict(method="replay", seed=0, per_class=pc, tag=f"replay_pc{pc}_s0"))
    for lam in (100.0, 10000.0):
        out.append(dict(method="ewc", seed=0, ewc_lambda=lam, tag=f"ewc_lam{int(lam)}_s0"))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "10")))
    jobs = {"main": jobs_main, "ablation": jobs_ablation}[what]()
    mine = [j for k, j in enumerate(jobs) if k % nshard == shard]

    print(f"[{what}] shard {shard}/{nshard} -> {len(mine)} jobs", flush=True)
    t0 = time.time()
    for k, j in enumerate(mine):
        print(f"  [{k+1}/{len(mine)}] {j}", flush=True)
        run_one(j, verbose=True)
    print(f"done in {time.time()-t0:.1f}s", flush=True)
