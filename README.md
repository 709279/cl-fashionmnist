# Split-Fashion-MNIST 持续学习实验

深度学习课程设计。在 Split-Fashion-MNIST 的五任务序列上对比四种抗遗忘策略，
并用联合训练（Joint Training）作为性能上限参考。

## 实验设置

- **数据集**：Fashion-MNIST，按类别切分为 5 个任务（每任务 2 类）
- **模型**：小型 CNN（width=32，dropout=0.3）
- **训练**：AdamW，lr=1e-3，wd=5e-4，batch=512，每任务 3 epoch，余弦退火
- **随机种子**：0 / 1

## 对比方法

| 方法 | 说明 |
| --- | --- |
| Naive（顺序微调） | 无任何抗遗忘机制，作为灾难性遗忘的基线 |
| Replay（经验回放） | 每类保留 40 个样本的回放缓冲区 |
| EWC | Fisher 信息对角线加权的参数正则，λ=1000 |
| LwF | 旧模型软标签蒸馏，α=1.0，T=2.0 |
| Joint（联合训练） | 一次性使用全部任务数据，作为上限 |

## 评价指标

- **ACC**：学完所有任务后，各任务测试准确率的平均值
- **BWT**（Backward Transfer）：学习新任务对旧任务性能的平均影响
- **Forgetting**：各任务历史最优准确率与最终准确率的平均落差

## 目录结构

```
.
├── scripts/
│   ├── run_cl.py            # 实验调度入口（支持分片并行）
│   ├── make_cl_figures.py   # 由 results_cl/runs/*.json 绘制论文级图表
│   └── build_cl_report.py   # 由实验结果自动生成 Markdown / DOCX 报告
├── src/
│   ├── common.py            # 随机种子、目录、结果序列化
│   ├── data.py              # Fashion-MNIST 加载、Dataset 封装、批增强
│   ├── models.py            # 小型 CNN
│   ├── cl_data.py           # 任务切分与准确率矩阵评估
│   └── cl_methods.py        # Replay / EWC / LwF 与统一训练引擎
├── docs/                    # 课程设计任务书
└── report/                  # 最终提交报告
```

## 运行

```bash
pip install -r requirements.txt

# 主实验：4 种方法 × 2 个种子，外加 1 次联合训练
python scripts/run_cl.py main 0 1

# 消融实验：回放容量、EWC 正则强度
python scripts/run_cl.py ablation 0 1

# 出图 + 生成报告
python scripts/make_cl_figures.py
python scripts/build_cl_report.py
```

`run_cl.py` 的后两个参数是「第几片 / 共几片」，可以把任务拆到多台机器上跑：

```bash
python scripts/run_cl.py main 0 2   # 机器 A
python scripts/run_cl.py main 1 2   # 机器 B
```

结果以 JSON 保存在 `results_cl/runs/cl_<tag>.json`，报告中的所有数字都由脚本
从这些 JSON 读取，避免手工誊抄出错。
