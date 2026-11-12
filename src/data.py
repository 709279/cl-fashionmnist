"""长尾数据集构造与快速数据管线。

按照 Cui et al. (CVPR 2019) 的指数衰减方式，从平衡数据集构造人工长尾分布：
    n_i = n_max * (1 / IR) ** (i / (C - 1))
其中 i 为按样本数降序排列的类别序号，IR = n_max / n_min 为不平衡比。

为在 CPU 上高效训练，数据集整体以 uint8 张量驻留内存，数据增强
（随机裁剪 + 水平翻转）在整批张量上完成，避免逐张 PIL 变换的开销。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets

from common import DATA_DIR

RAW_DIR = os.path.join(DATA_DIR, "raw")


class ArrayDataset(Dataset):
    """直接索引内存中的 uint8 张量。"""

    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return self.images[j], int(self.labels[j])


class BatchAugment:
    """把一批 uint8 图像转成归一化 float 张量，并施加随机裁剪 + 翻转。"""

    def __init__(self, mean, std, train=True, pad=4):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        self.train = train
        self.pad = pad

    def __call__(self, batch):
        xs = torch.stack([b[0] for b in batch]).float().div_(255.0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        if self.train:
            if self.pad > 0:
                xs = F.pad(xs, (self.pad,) * 4, mode="reflect")
            h, w = xs.shape[-2:]
            out = torch.empty(xs.shape[0], xs.shape[1], h - 2 * self.pad, w - 2 * self.pad)
            for k in range(xs.shape[0]):
                i = torch.randint(0, 2 * self.pad + 1, (1,)).item()
                j = torch.randint(0, 2 * self.pad + 1, (1,)).item()
                out[k] = xs[k, :, i:i + h - 2 * self.pad, j:j + w - 2 * self.pad]
            xs = out
            flip = torch.rand(xs.shape[0]) < 0.5
            xs[flip] = torch.flip(xs[flip], dims=[-1])
        xs = (xs - self.mean) / self.std
        return xs, ys


def load_base(name, train):
    if name == "fashion":
        ds = datasets.FashionMNIST(root=RAW_DIR, train=train, download=False)
        mean, std = (0.2860,), (0.3530,)
    elif name == "cifar10":
        ds = datasets.CIFAR10(root=RAW_DIR, train=train, download=True)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError(name)
    imgs = ds.data
    if imgs.ndim == 3:  # HWC 灰度图
        imgs = imgs[:, None, :, :]
    else:  # HWC 彩色图 -> CHW
        imgs = imgs.transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(imgs)), torch.tensor(ds.targets), mean, std


def make_long_tail(labels, num_classes, imbalance_ratio, rng):
    labels = np.asarray(labels)
    counts_per_class = np.array([(labels == c).sum() for c in range(num_classes)])
    order = np.argsort(-counts_per_class)  # 类别 0 为头部类
    n_max = int(counts_per_class[order[0]])
    n_min = max(1, int(round(n_max / imbalance_ratio)))
    num_per_class = {}
    for rank, c in enumerate(order):
        n_c = int(round(n_max * (imbalance_ratio ** (-rank / (num_classes - 1)))))
        num_per_class[int(c)] = max(n_min, n_c)

    indices = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        indices.extend(idx_c[: num_per_class[c]].tolist())
    indices = np.array(indices)
    rng.shuffle(indices)
    return indices, num_per_class


def make_balanced_subset(labels, num_classes, rng):
    labels = np.asarray(labels)
    counts = np.array([(labels == c).sum() for c in range(num_classes)])
    m = int(counts.min())
    idx = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        idx.extend(idx_c[:m].tolist())
    return np.array(idx)


def build_loaders(cfg, seed):
    name = cfg["dataset"]
    rng = np.random.RandomState(seed + 1000)
    tr_imgs, tr_labels, mean, std = load_base(name, True)
    te_imgs, te_labels, _, _ = load_base(name, False)
    tr_labels_np = tr_labels.numpy()

    num_classes = cfg["num_classes"]
    lt_idx, num_per_class = make_long_tail(
        tr_labels_np, num_classes, cfg["imbalance_ratio"], rng)

    # 从长尾训练集划出 5% 作为监控用验证集
    rng2 = np.random.RandomState(seed + 2000)
    perm = rng2.permutation(len(lt_idx))
    n_val = int(0.05 * len(lt_idx))
    val_idx, tr_idx = lt_idx[perm[:n_val]], lt_idx[perm[n_val:]]

    aug_train = BatchAugment(mean, std, train=True)
    aug_eval = BatchAugment(mean, std, train=False)

    train_set = ArrayDataset(tr_imgs, tr_labels, tr_idx)
    val_set = ArrayDataset(tr_imgs, tr_labels, val_idx)
    test_set = ArrayDataset(te_imgs, te_labels, np.arange(len(te_labels)))

    labels_tr = tr_labels_np[tr_idx]
    counts = np.array([(labels_tr == c).sum() for c in range(num_classes)])
    class_counts = {int(c): int(counts[c]) for c in range(num_classes)}

    bs = cfg["batch_size"]
    if cfg.get("sampler") == "balanced":
        w = 1.0 / np.maximum(counts, 1).astype(np.float64)
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(w[labels_tr]), num_samples=len(tr_idx), replacement=True)
        train_loader = DataLoader(train_set, batch_size=bs, sampler=sampler,
                                  num_workers=0, drop_last=True,
                                  collate_fn=aug_train)
    else:
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                                  num_workers=0, drop_last=True,
                                  collate_fn=aug_train)

    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=0,
                            collate_fn=aug_eval)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0,
                             collate_fn=aug_eval)

    bal_idx = make_balanced_subset(labels_tr, num_classes, rng)
    bal_set = ArrayDataset(tr_imgs, tr_labels, tr_idx[bal_idx])
    bal_loader = DataLoader(bal_set, batch_size=min(64, bs), shuffle=True, num_workers=0,
                            drop_last=True, collate_fn=aug_train)

    meta = {
        "dataset": name,
        "imbalance_ratio": cfg["imbalance_ratio"],
        "num_per_class": {str(k): int(v) for k, v in sorted(num_per_class.items())},
        "class_counts_train": {str(k): v for k, v in sorted(class_counts.items())},
        "n_train": int(len(train_set)),
        "n_val": int(len(val_set)),
        "n_test": int(len(test_set)),
        "n_balanced": int(len(bal_set)),
    }
    return dict(train=train_loader, val=val_loader, test=test_loader,
                balanced=bal_loader, meta=meta, class_counts=counts)
