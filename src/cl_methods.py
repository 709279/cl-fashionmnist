"""持续学习的四种策略与训练引擎。

- naive : 顺序微调，只用当前任务数据（基线，遗忘最严重）
- replay: 每个旧任务缓存少量样本，与当前批混合训练
- ewc   : 用 Fisher 信息对重要参数施加二次惩罚，抑制漂移
- lwf   : 对旧模型在**当前任务数据**上的软输出做知识蒸馏，无需旧数据
- joint : 联合训练全部数据（上限参照，不参与排名）
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 经验回放 ----------------

class ReplayBuffer:
    """按类均衡缓存旧任务样本（uint8 原图，训练时再增强）。"""

    def __init__(self, per_class, num_classes=10, seed=0):
        self.per_class = per_class
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed + 500)
        self.data = {}          # class -> list of uint8 tensors
        self.labels = []

    def add_task(self, images, labels):
        for c in range(self.num_classes):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            self.rng.shuffle(idx)
            for j in idx[: self.per_class]:
                self.data.setdefault(c, []).append(images[j].clone())

    def __len__(self):
        return sum(len(v) for v in self.data.values())

    def sample(self, n, device=None):
        pool = [(c, img) for c, v in self.data.items() for img in v]
        if not pool:
            return None, None
        pick = self.rng.randint(0, len(pool), size=n)
        xs = torch.stack([pool[i][1] for i in pick]).float().div_(255.0)
        ys = torch.tensor([pool[i][0] for i in pick], dtype=torch.long)
        return xs, ys


# ---------------- EWC ----------------

def fisher_diagonal(model, loader, device, n_batches=4, seed=0):
    """经验 Fisher 信息（梯度的平方），只需少量 batch 即可稳定。"""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n_seen = 0
    for k, (x, y) in enumerate(loader):
        if k >= n_batches:
            break
        model.zero_grad(set_to_none=True)
        out = model(x.to(device))
        loss = F.cross_entropy(out, y.to(device), reduction="sum")
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += p.grad.detach().pow(2)
        n_seen += y.numel()
    for n in fisher:
        fisher[n] /= max(n_seen, 1)
    model.zero_grad(set_to_none=True)
    return fisher


def ewc_penalty(model, terms, lam):
    """terms: [(fisher_dict, star_params_dict), ...]"""
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    pen = 0.0
    for fisher, star in terms:
        for n, p in model.named_parameters():
            if n in fisher:
                pen = pen + (fisher[n].to(p.device) * (p - star[n].to(p.device)).pow(2)).sum()
    return 0.5 * lam * pen


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ---------------- LwF ----------------

def kd_loss(student_logits, teacher_logits, T=2.0):
    """教师软标签与学生软标签之间的 KL 散度（乘以 T^2 保持梯度量级）。"""
    p_t = F.softmax(teacher_logits / T, dim=1)
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


# ---------------- 训练一个任务 ----------------

def train_task(model, loader, method, opt, epochs, device, state, sched=None):
    """在单个任务上训练若干 epoch，返回平均训练损失。

    state 中携带方法相关的对象：
        replay_buffer / ewc_terms / ewc_lambda / old_model / lwf_alpha / T
    """
    ce = nn.CrossEntropyLoss()
    losses = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out, y)

            if method == "replay" and state.get("replay_buffer") is not None:
                buf = state["replay_buffer"]
                if len(buf) > 0:
                    xr, yr = buf.sample(x.size(0))
                    if xr is not None:
                        # 缓存样本不做随机裁剪/翻转，只做与训练集一致的归一化
                        xr = (xr.to(device) - state["mean"]) / state["std"]
                        loss = loss + ce(model(xr), yr.to(device))

            if method == "ewc":
                loss = loss + ewc_penalty(model, state.get("ewc_terms", []),
                                          state.get("ewc_lambda", 1000.0))

            if method == "lwf" and state.get("old_model") is not None:
                with torch.no_grad():
                    t_out = state["old_model"](x)
                loss = loss + state.get("lwf_alpha", 1.0) * kd_loss(out, t_out,
                                                                    state.get("lwf_T", 2.0))

            loss.backward()
            opt.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
        losses.append(tot / max(n, 1))
        if sched is not None:
            sched.step()
    return losses


def clone_for_teacher(model):
    m = copy.deepcopy(model)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
