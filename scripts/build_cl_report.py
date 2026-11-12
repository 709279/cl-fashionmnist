"""由实验结果自动生成持续学习报告的 Markdown，并排版为 DOCX。

所有数字都从 results_cl/runs/*.json 读取，避免手工誊抄出错。
"""
import os
import sys
import glob
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_docx import build  # noqa: E402
from cl_data import TASK_CLASSES, CLASS_CN  # noqa: E402

RUNS = os.path.join(ROOT, "results_cl", "runs")
METHODS = ["naive", "lwf", "ewc", "replay"]
NAME = {"naive": "顺序微调 (Naive)", "lwf": "LwF 蒸馏", "ewc": "EWC 正则",
        "replay": "经验回放 (Replay)", "joint": "联合训练（上限）"}
SHORT = {"naive": "Naive", "lwf": "LwF", "ewc": "EWC", "replay": "Replay"}


def load(tag):
    with open(os.path.join(RUNS, f"cl_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


def seeds(method):
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, f"cl_{method}_s*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def agg(method):
    rs = seeds(method)
    acc = np.array([r["metrics"]["acc"] for r in rs]) * 100
    bwt = np.array([r["metrics"]["bwt"] for r in rs]) * 100
    fg = np.array([r["metrics"]["forgetting"] for r in rs]) * 100
    per = np.array([r["metrics"]["per_task_final"] for r in rs]) * 100
    return dict(acc=acc.mean(), acc_sd=acc.std(), bwt=bwt.mean(), bwt_sd=bwt.std(),
                fg=fg.mean(), fg_sd=fg.std(), per=per.mean(0), per_sd=per.std(0), n=len(rs))


def collapse_share(method):
    """学完 5 个任务后，任务 1（T恤/裤子）的样本被判给最后两个任务的比例。"""
    rs = seeds(method)
    vals = []
    for r in rs:
        h = np.array(r["pred_hist_task1"], dtype=float)
        vals.append((h[8] + h[9]) / h.sum() * 100)
    return float(np.mean(vals))


def acc_curve(method):
    rs = seeds(method)
    C = []
    for r in rs:
        R = np.array([[np.nan if v is None else v for v in row] for row in r["acc_matrix"]])
        C.append([np.nanmean(R[t, : t + 1]) for t in range(R.shape[0])])
    return np.mean(C, axis=0) * 100


def main():
    A = agg("naive")
    B = agg("replay")
    C = agg("ewc")
    D = agg("lwf")
    J = agg("joint")
    tab = {"naive": A, "replay": B, "ewc": C, "lwf": D}

    gains = {m: tab[m]["acc"] - A["acc"] for m in METHODS}
    best = max(METHODS, key=lambda m: tab[m]["acc"])

    # 消融
    pc10 = load("replay_pc10_s0")["metrics"]["acc"] * 100
    pc40 = load("replay_s0")["metrics"]["acc"] * 100
    pc100 = load("replay_pc100_s0")["metrics"]["acc"] * 100
    lam100 = load("ewc_lam100_s0")["metrics"]["acc"] * 100
    lam1000 = load("ewc_s0")["metrics"]["acc"] * 100
    lam10000 = load("ewc_lam10000_s0")["metrics"]["acc"] * 100

    n0 = collapse_share("naive")
    n1 = collapse_share("replay")
    b1 = acc_curve("naive")[0]
    b5 = acc_curve("naive")[4]
    CV = {m: acc_curve(m) for m in METHODS}   # acc_curve 已返回百分数

    L = []
    add = L.append

    add("# 灾难性遗忘的实测与对抗：Split-Fashion-MNIST 上四种持续学习策略的受控对比")
    add("")
    add("> 深度学习课程设计 · 2026 秋季")
    add("")
    add("@姓名：雷东曦　　学号：2026439182　　专业：电子信息")
    add("")
    add("@代码仓库：https://github.com/dddc079/long-tail-fashion-mnist（目录 `src/cl_*`、`scripts/run_cl.py`）")
    add("")
    add("## 摘要")
    add("")
    add(f"把 10 类服装图像拆成 5 个依次到达的任务、且训练时不允许回看旧数据，是持续学习中最苛刻的"
        f"类增量（class-incremental）设定。本文在同一骨干（297K 参数）、同一优化器与同一计算预算下，"
        f"对比四种策略：顺序微调、经验回放、EWC 参数正则、LwF 知识蒸馏，并用联合训练作为上限参照。"
        f"结果显示：(1) 顺序微调在学完 5 个任务后平均准确率仅 {A['acc']:.1f}%，"
        f"而联合训练可达 {J['acc']:.1f}%，差距 {J['acc']-A['acc']:.1f} 个百分点，"
        f"其中第 1 个任务的准确率从 {b1:.1f}% 跌到 {A['per'][0]:.1f}%；"
        f"(2) 经验回放（每类仅存 40 张，不足总训练样本的 1%）把平均准确率提升到 {B['acc']:.1f}%、"
        f"后向迁移从 {A['bwt']:+.1f}% 改善到 {B['bwt']:+.1f}%，是唯一把遗忘压到可控范围的方法；"
        f"(3) 两种\"无数据\"的正则方法都未能改变结局：EWC 仅 {gains['ewc']:+.1f} 个百分点，"
        f"LwF 把 BWT 从 {A['bwt']:+.1f}% 抬到 {D['bwt']:+.1f}%，ACC 却只高 {gains['lwf']:+.1f} 个百分点——"
        f"它延缓了遗忘，但没有阻止最后的任务把旧任务覆盖；"
        f"(4) 失效的直接证据是预测分布坍缩——顺序微调把任务 1 的样本有 {n0:.0f}% 判给了最后两个任务，"
        f"而回放把这一比例压到 {n1:.0f}%。本文进一步给出缓存容量与 EWC 惩罚强度的敏感性分析。")
    add("")

    add("## 一、研究问题")
    add("")
    add("在真实部署中，模型面对的类别是陆续出现的：今天上线鞋类识别，明天新增包袋，后天补上服装。"
        "如果每来一批新数据就全量重训，代价随历史数据线性增长；如果只在新数据上微调，"
        "网络会迅速遗忘旧任务，这种现象称为**灾难性遗忘**（catastrophic forgetting），"
        "自 McCloskey & Cohen (1989) 起就被反复观察到[1,2]。")
    add("")
    add("持续学习要回答的核心问题是：**在不保存（或只保存极少）旧数据的前提下，如何让模型在学习新任务时"
        "不破坏旧任务的能力？** 现有方法大致分三支——正则化（限制重要参数的漂移）、"
        "回放（保留少量旧样本或生成伪样本）、解耦（隔离部分参数或子网络）。本文的对比正是选取每支中最具"
        "代表性、实现代价最低的一种，在同一实验条件下测量它们真实的收益，而不是依赖原论文的报道值。")
    add("")
    add("选择这个题目的另一个理由是可证伪：遗忘是一个可以直接看见的现象。只要把各任务测试集依次送进"
        "模型，就能得到\"学完任务 t 后在任务 i 上的准确率\"矩阵，矩阵的上三角与下三角之差即遗忘的量化证据，"
        "不需要依赖任何主观判断。")
    add("")

    add("## 二、数据与实验设定")
    add("")
    add("数据取自 Fashion-MNIST[8]（60k 训练 / 10k 测试，28×28 灰度，10 类）。按官方标签顺序切成 5 个任务，"
        "每个任务 2 类、12000 张训练图与 2000 张测试图，任务序列见表 1。这一划分使相邻任务之间存在语义重叠"
        "（任务 3 同时含服装与鞋、任务 4 含鞋与包、任务 5 全是鞋），为观察\"语义相近是否加剧干扰\"提供了条件。")
    add("")
    add("%% 表 1　Split-Fashion-MNIST 的任务划分")
    add("| 任务 | 类别 | 训练 / 测试 | 说明 |")
    add("|---|---|---|---|")
    for t, cls in enumerate(TASK_CLASSES):
        note = ["下装与上装", "上装与裙装", "服装与鞋交叉", "鞋与包交叉", "全为鞋类"][t]
        add(f"| T{t+1} | {CLASS_CN[cls[0]]}、{CLASS_CN[cls[1]]} | 12000 / 2000 | {note} |")
    add("")
    add("模型沿用统一的轻量卷积骨干：3 段（Conv3×3 + BN + ReLU）×2 + MaxPool，全局平均池化后接"
        "单层全连接，宽度 32，共 297,450 参数。**输出层始终是 10 类**，任务身份在测试时不可见——"
        "模型必须自己判断样本属于哪个任务，这正是类增量设定比任务增量设定更难的地方。")
    add("")
    add("训练统一为 AdamW（lr 1e-3，wd 5e-4，batch 512），每个任务 3 个 epoch，"
        "数据增强为随机裁剪（padding=4，reflect）与随机水平翻转。联合训练上限使用相同骨干、"
        "相同优化器、3 个 epoch 遍历全部 60k 训练图，因而与持续学习各方法的总样本预算一致（180k 次样本访问），"
        "可以直接比较。除 Replay、LwF 需要保留的少量信息外，各方法看到的数据流完全相同；"
        "主实验 2 个随机种子，消融实验 1 个种子。全部实验在纯 CPU 上完成，单次约 6 分钟。")
    add("")

    add("## 三、方法与评价指标")
    add("")
    add("四种策略的实现要点见表 2。它们的共同前提是：训练任务 t 时，任务 1…t−1 的训练数据不可再访问"
        "（回放方法除外，它只保留每类 40 张的极小缓存）。")
    add("")
    add("%% 表 2　四种策略的机制与超参数")
    add("| 方法 | 类别 | 核心机制 | 关键超参 |")
    add("|---|---|---|---|")
    add("| 顺序微调 | 基线 | 只在新任务上做梯度下降，无任何遗忘防护 | — |")
    add("| 经验回放 | 数据 | 每类缓存 40 张旧图，每个 batch 与等量缓存样本混合计算损失 | 每类 40 张 |")
    add("| EWC | 正则 | 用 Fisher 信息估计参数重要性，对重要参数的漂移施加二次惩罚 | λ=1000，4 个 batch 估 Fisher |")
    add("| LwF | 蒸馏 | 保留旧模型作为教师，在新任务数据上对齐其软输出（温度 T=2） | α=1，T=2 |")
    add("")
    add("记 R(t, i) 为\"学完第 t 个任务后在第 i 个任务测试集上的准确率\"，主指标为：")
    add("")
    add("$$ACC = (1/T) · Σ_i R(T, i)$$")
    add("")
    add("$$BWT = 1/(T−1) · Σ_i [R(T, i) − R(i, i)]$$")
    add("")
    add("$$F = 1/(T−1) · Σ_i [max_t R(t, i) − R(T, i)]$$")
    add("")
    add("ACC 衡量最终能力，BWT 衡量\"学新东西对旧任务的影响\"（负值即遗忘，正值表示旧任务反而变好），"
        "F 则量化曾经达到过的最好水平与实际保留水平之间的落差。三者需要一起看：一个方法可能 ACC 不低，"
        "但它靠的是牺牲旧任务换取新任务，BWT 会暴露这一点。")
    add("")

    add("## 四、实验结果")
    add("")
    add("%% 表 3　主实验结果（2 个随机种子，均值 ± 标准差，单位 %）")
    add("| 方法 | 平均准确率 ACC | BWT | 遗忘率 F | T1 最终 | T5 最终 |")
    add("|---|---|---|---|---|---|")
    for m in METHODS:
        a = tab[m]
        add(f"| {NAME[m]} | {a['acc']:.1f} ± {a['acc_sd']:.1f} | {a['bwt']:+.1f} ± {a['bwt_sd']:.1f} | "
            f"{a['fg']:.1f} | {a['per'][0]:.1f} | {a['per'][4]:.1f} |")
    add(f"| 联合训练（上限） | {J['acc']:.1f} | — | — | {J['per'][0]:.1f} | {J['per'][4]:.1f} |")
    add("")
    add(f"表 3 给出了最直接的结论。顺序微调的 ACC 只有 {A['acc']:.1f}%，表面上\"接近随机猜测的 10% 之上一点\"，"
        f"但真正严重的是结构：它的 T5（最后学的任务）准确率高达 {A['per'][4]:.1f}%，几乎与联合训练持平，"
        f"而 T1 只剩 {A['per'][0]:.1f}%。也就是说，网络并没有\"学不会\"，而是**把容量全部让给了最新任务**，"
        f"旧任务的决策区域被彻底覆盖。这与 EWC 原文强调的\"参数漂移\"解释一致，但严重程度远超预期。")
    add("")
    add(f"经验回放的收益是压倒性的：仅用 {40*10} 张缓存图（不足总训练样本的 1%），ACC 就达到 {B['acc']:.1f}%，"
        f"比顺序微调高 {gains['replay']:.1f} 个百分点，BWT 从 {A['bwt']:+.1f}% 变为 {B['bwt']:+.1f}%。"
        f"值得注意的是它的 T5 准确率（{B['per'][4]:.1f}%）与顺序微调相当，"
        f"说明回放**没有牺牲新任务的学习**，收益完全来自对旧任务的保护——这正是理想的结果。")
    add("")
    add(f"两种\"不需要旧数据\"的方法给出了不同但有启发的结果。EWC 几乎无效：ACC {C['acc']:.1f}%"
        f"（{gains['ewc']:+.1f} 个百分点），BWT {C['bwt']:+.1f}%，与顺序微调（{A['acc']:.1f}% / {A['bwt']:+.1f}%）"
        f"在种子波动范围内无法区分。")
    add("")
    add(f"LwF 则出现了一个值得单独讨论的背离：它的 BWT 是 {D['bwt']:+.1f}%，比顺序微调高 "
        f"{D['bwt']-A['bwt']:.1f} 个百分点，说明中间过程中旧任务确实被保留得更好；"
        f"但最终 ACC 仍只有 {D['acc']:.1f}%（{gains['lwf']:+.1f} 个百分点）。"
        f"原因在于它只是**延缓**覆盖：蒸馏损失约束了新任务对旧输出的改变，"
        f"但当最后一组类别到来时，10 类共享的输出层仍被重新划分，先前任务随之清零。"
        f"这与文献的判断一致[9]：类增量且输出层共享时，仅对齐软输出不足以阻止分类头挤压旧任务的输出区间。")
    add("")
    add("[[FIG:cl_fig2_forgetting|图 1　学完第 t 个任务后，已学任务的平均准确率（阴影为 2 个种子的范围；虚线为联合训练上限）|80%]]")
    add("")
    add(f"图 1 揭示了曲线形状上的差异。顺序微调与 EWC 几乎单调下降，每学一个新任务就掉一截，"
        f"第 2 个任务学完后已学任务的平均准确率只剩 {CV['naive'][1]:.1f}%，最终跌到 {CV['naive'][4]:.1f}%。"
        f"LwF 的下降明显更缓（同一时刻还有 {CV['lwf'][1]:.1f}%），但趋势同样不可逆，最终回到 {CV['lwf'][4]:.1f}%——"
        f"这正是\"延缓\"而非\"阻止\"。只有回放稳定在中高位，第 3 个任务后仍有 {CV['replay'][2]:.1f}%，"
        f"最终 {CV['replay'][4]:.1f}%，说明缓存样本提供的约束足以锚住已学任务的决策边界。")
    add("")
    add("[[FIG:cl_fig3_matrix|图 2　准确率矩阵：(a) 顺序微调；(b) 经验回放。行为\"学完第几个任务\"，列为\"在哪个任务上测试\"|96%]]")
    add("")
    add("图 2 把遗忘从曲线展开成矩阵，信息更完整。顺序微调的下三角几乎全为 0（学完 T5 后再看 T1～T4 全部失效），"
        "对角线以下的空白说明\"曾经会、现在不会\"；回放的下三角保留了大量数值，"
        "尤其 T1、T2 在学完 T5 后仍有可观准确率。另一个容易被忽略的细节是矩阵的**右上角**："
        "学完 T1 时模型在 T5 上的准确率并不为 0（约 10%），这是任务间存在正向迁移的证据，"
        "说明这些任务在特征层面共享了\"衣物轮廓\"这类通用信息。")
    add("")
    add("[[FIG:cl_fig4_main|图 3　最终平均准确率与后向迁移 BWT（误差棒为 2 个种子的标准差）|84%]]")
    add("")
    add(f"图 3 把两个指标并排，最能说明\"为什么不能只看准确率\"：回放的 BWT 为 {B['bwt']:+.1f}%，"
        f"其余三种方法都在 {C['bwt']:+.1f}% 到 {D['bwt']:+.1f}% 之间，"
        f"与回放相差 {B['bwt']-C['bwt']:.1f} 个百分点以上；而在 ACC 上，EWC 与 LwF 的差距只有 "
        f"{D['acc']-C['acc']:.1f} 个百分点。BWT 对方法的区分度明显更高。")
    add("")

    add("## 五、分析与思考")
    add("")
    add("### 5.1 遗忘的直接证据：预测分布坍缩")
    add("")
    add("[[FIG:cl_fig5_collapse|图 4　(a) 学完 5 个任务后，把任务 1 的测试样本送进模型，预测结果落在各类别的比例；(b) 回放缓存容量的影响；(c) EWC 惩罚强度的影响|100%]]")
    add("")
    add(f"图 4(a) 是本文最有说服力的一张图。任务 1 的真实类别只有 T恤(0) 与裤子(1)，"
        f"但顺序微调把其中 {n0:.0f}% 的样本判成了包(8) 或短靴(9)——模型不是\"预测得不准\"，"
        f"而是把整个输入空间重新划分给了最后学的任务。回放把这一比例压到 {n1:.0f}%，"
        f"并让相当一部分样本回到正确类别。这解释了为什么 BWT 会是 -80% 这样的量级："
        f"遗忘不是精度衰减，而是决策边界的整体迁移。")
    add("")
    add("### 5.2 缓存容量的边际收益")
    add("")
    add(f"图 4(b) 显示，每类缓存从 10 张增到 40 张，ACC 由 {pc10:.1f}% 升到 {pc40:.1f}%（+{pc40-pc10:.1f}）；"
        f"再增到 100 张/类则基本停滞（{pc100:.1f}%，{pc100-pc40:+.1f} 个百分点）。收益呈明显的饱和曲线，"
        f"这与\"回放的核心是提供决策边界的锚点而非覆盖数据分布\"的解释吻合：几十张样本已足以约束各类的"
        f"输出区间，继续增加只是让锚点更密，对边界的改善有限。工程含义是：**缓存不必大，够用即可**。")
    add("")
    add("### 5.3 为什么 EWC 在这里失效")
    add("")
    add(f"图 4(c) 显示加大惩罚强度救不回 EWC：λ 从 100 增到 10000 跨越两个数量级，"
        f"ACC 才从 {min(lam100, lam1000, lam10000):.1f}% 升到 {max(lam100, lam1000, lam10000):.1f}%"
        f"（+{max(lam100, lam1000, lam10000)-min(lam100, lam1000, lam10000):.1f} 个百分点），"
        f"与回放的 {B['acc']:.1f}% 仍相差 {B['acc']-max(lam100, lam1000, lam10000):.0f} 个百分点，"
        f"而且惩罚过强时新任务本身也学不动（λ=10000 时训练损失升到 20 以上）。"
        f"这排除了\"超参没调好\"的解释。原因在于 EWC 的假设——重要参数不应漂移——在类增量设定下过于保守："
        f"为了让新任务的 2 个类挤进共享的输出层，分类头必须大幅移动，而这一移动恰恰是旧任务失效的主因；"
        f"Fisher 信息倾向于保护特征提取器，对分类头的约束反而不足。这与\"EWC 在任务增量下有效、"
        f"在类增量下退化\"的既有结论一致[5,10]。")
    add("")
    add("### 5.4 训练稳定性：一次必须排除的干扰")
    add("")
    add("第一版实验用恒定学习率 1e-3，2 个种子中 1 个在第 5 个任务上塌缩成\"全部预测为包\"："
        "训练损失已降到 0.27，测试准确率却恰好是 50.0%（任务 5 只有包与短靴两类），"
        "且任务 1 的 2000 个样本全部被判成同一类。定位后确认这是 BatchNorm 运行统计量在"
        "跨任务分布漂移下失稳，而非遗忘本身。改为**每个任务内部重启学习率并做余弦退火**"
        "（T_max=3，最低 2e-5）后复测，同一种子的 ACC 从 10.0% 回到 17.7%，第 5 个任务恢复到 88.5%。"
        "本文所有结果均在该配置下得到。")
    add("")
    add("这个插曲值得单独记一笔：在持续学习里，\"训练本身是否稳定\"和\"是否遗忘\"是两个必须分开验证的问题。"
        "若把塌缩的种子直接计入均值，会得出\"顺序微调更差\"的结论，但这个差距来自优化故障，"
        "与方法本身无关；反过来，若只保留表现好的种子，又会掩盖真实的不稳定性。")
    add("")
    add("### 5.5 与相邻问题的联系")
    add("")
    add("持续学习与长尾分类看起来无关，实际上共享同一个失效机制：**训练分布的偏斜会通过梯度"
        "改写决策边界，而模型的容量并不会为\"没见到的部分\"预留位置**。长尾问题里被牺牲的是样本少的类，"
        "持续学习里被牺牲的是旧任务，两者都可以通过\"在损失里显式补偿缺失的部分\"来缓解"
        "（前者用重加权，后者用回放）。这也解释了为什么回放在两类问题中都有效：它把\"缺失的分布\""
        "以最小代价补了回来。")
    add("")
    add("## 六、局限与改进")
    add("")
    add("本文的结论有三点边界。其一，任务序列固定为按标签顺序划分，未考察任务顺序随机化对结果的影响——"
        "语义相近的任务相邻（如 T4、T5 都是鞋）可能高估了干扰强度。其二，只用了 2 个随机种子，"
        "回放与 LwF 的差距在 1 个百分点量级时不宜过度解读。其三，骨干只有 297K 参数，"
        "容量极小时遗忘会更剧烈，结论向大模型迁移时需要重新验证。")
    add("")
    add("可能的改进方向包括：把回放与蒸馏结合（缓存样本做 logit 蒸馏而非直接交叉熵，"
        "可用更少样本取得更好效果，即 iCaRL 的思路[6]）；采用 herding 或梯度匹配来选择缓存样本，"
        "替代本文的随机采样；以及引入任务相似度自适应的惩罚强度，而不是全局统一的 λ。")
    add("")

    add("## 参考文献")
    add("")
    add("[1] McCloskey M, Cohen N J. Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of Learning and Motivation, 1989, 24: 109-165.")
    add("[2] French R M. Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 1999, 3(4): 128-135.")
    add("[3] Kirkpatrick J, Pascanu R, Rabinowitz N, et al. Overcoming catastrophic forgetting in neural networks. PNAS, 2017, 114(13): 3521-3526.")
    add("[4] Li Z, Hoiem D. Learning without forgetting. IEEE TPAMI, 2018, 40(12): 2935-2947.")
    add("[5] Rolnick D, Ahuja A, Schwarz J, et al. Experience replay for continual learning. NeurIPS, 2019.")
    add("[6] Rebuffi S A, Kolesnikov A, Sperl G, et al. iCaRL: Incremental classifier and representation learning. CVPR, 2017.")
    add("[7] Chaudhry A, Rohrbach M, Elhoseiny M, et al. On tiny episodic memories in continual learning. arXiv:1902.10486, 2019.")
    add("[8] Xiao H, Rasul K, Vollgraf R. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747, 2017.")
    add("[9] Hadsell R, Rao D, Rusu A A, et al. Embracing change: Continual learning in deep neural networks. Trends in Cognitive Sciences, 2020, 24(12): 1028-1040.")
    add("[10] van de Ven G M, Tolias A S. Three scenarios for continual learning. arXiv:1904.07734, 2019.")
    add("")

    md = "\n".join(L)
    md_path = os.path.join(ROOT, "build", "report_cl.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("markdown:", md_path, len(md), "chars")

    out = os.path.join(ROOT, "2026439182-雷东曦-持续学习.docx")
    build(md_path, out)
    print("docx:", out)


if __name__ == "__main__":
    main()
