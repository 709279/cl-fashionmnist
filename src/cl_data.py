"""Split-Fashion-MNIST：把 10 类拆成 5 个依次到来的任务（类增量设定）。

每个任务只包含 2 个类，模型在任务 t 上训练时**看不到**之前任务的数据
（Replay / LwF 除外，它们分别用少量样本缓存和蒸馏来对抗遗忘）。
测试时模型要同时回答全部 5 个任务的样本，但不知道样本属于哪个任务，
因此输出层始终是 10 类，遗忘会直接体现在预测分布上。
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ArrayDataset, BatchAugment, load_base

# 5 个任务的类别划分（按 Fashion-MNIST 官方标签顺序）
TASK_CLASSES = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

CLASS_CN = ["T恤", "裤子", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]

# 任务语义分组，用于分析「相邻任务语义相近是否加剧遗忘」
TASK_SEMANTIC = ["上装 A", "上装 B", "鞋服混", "鞋包混", "全鞋"]


def _task_index(labels, classes):
    idx = []
    for c in classes:
        idx.extend(np.where(labels == c)[0].tolist())
    return np.array(idx)


def build_tasks(cfg, seed):
    """返回每任务的训练/测试 loader 以及元信息。"""
    tr_imgs, tr_labels, mean, std = load_base("fashion", True)
    te_imgs, te_labels, _, _ = load_base("fashion", False)
    labels_tr = tr_labels.numpy()
    labels_te = te_labels.numpy()

    aug_tr = BatchAugment(mean, std, train=True)
    aug_ev = BatchAugment(mean, std, train=False)

    rng = np.random.RandomState(seed + 7000)
    per_task = cfg.get("per_task", 0)          # 0 表示用完整数据
    bs = cfg["batch_size"]

    train_loaders, test_loaders, meta_tasks, raw_tr = [], [], [], []
    for t, cls in enumerate(TASK_CLASSES):
        idx_tr = _task_index(labels_tr, cls)
        idx_te = _task_index(labels_te, cls)
        if per_task and len(idx_tr) > per_task:
            idx_tr = np.sort(rng.permutation(idx_tr)[:per_task])
        train_loaders.append(DataLoader(
            ArrayDataset(tr_imgs, tr_labels, idx_tr), batch_size=bs, shuffle=True,
            num_workers=0, drop_last=True, collate_fn=aug_tr))
        test_loaders.append(DataLoader(
            ArrayDataset(te_imgs, te_labels, idx_te), batch_size=512, shuffle=False,
            num_workers=0, collate_fn=aug_ev))
        raw_tr.append((tr_imgs[idx_tr], tr_labels[idx_tr]))
        meta_tasks.append({
            "task": t,
            "classes": list(cls),
            "class_names": [CLASS_CN[c] for c in cls],
            "n_train": int(len(idx_tr)),
            "n_test": int(len(idx_te)),
        })

    # 联合训练（上限参照）：一次看全部数据，同样 3 个 epoch 的样本预算
    all_idx = np.arange(len(labels_tr))
    joint_loader = DataLoader(
        ArrayDataset(tr_imgs, tr_labels, all_idx), batch_size=bs, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=aug_tr)

    meta = {
        "dataset": "fashion",
        "n_tasks": len(TASK_CLASSES),
        "tasks": meta_tasks,
        "n_train_total": int(len(labels_tr)),
        "joint_batches_per_epoch": len(joint_loader),
    }
    return dict(train=train_loaders, test=test_loaders, joint=joint_loader,
                raw=raw_tr, meta=meta)


@torch.no_grad()
def task_accuracy(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        n += y.size(0)
    return correct / max(n, 1)


@torch.no_grad()
def eval_matrix_row(model, test_loaders, device, upto):
    """学完任务 upto 后，在**全部**任务上的准确率（未学的任务也测，用于观察迁移）。"""
    return [float(task_accuracy(model, ld, device)) for ld in test_loaders]


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()
