"""轻量卷积网络（面向 28x28 / 32x32 小图，CPU 可训练）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """三段式卷积网络，末段全局平均池化后接线性分类器。

    约 0.1M 参数，保证在 CPU 上可以完成多组对比实验。
    """

    def __init__(self, in_ch=1, num_classes=10, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.stem = conv_bn(in_ch, w)
        self.b1 = nn.Sequential(conv_bn(w, w), conv_bn(w, w), nn.MaxPool2d(2))
        self.b2 = nn.Sequential(conv_bn(w, 2 * w), conv_bn(2 * w, 2 * w), nn.MaxPool2d(2))
        self.b3 = nn.Sequential(conv_bn(2 * w, 4 * w), conv_bn(4 * w, 4 * w), nn.MaxPool2d(2))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * w, num_classes)
        self.out_dim = 4 * w

    def features(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def forward(self, x):
        h = self.features(x)
        return self.fc(self.drop(h))


def build_model(cfg):
    in_ch = 1 if cfg["dataset"] == "fashion" else 3
    return SmallCNN(in_ch=in_ch, num_classes=cfg["num_classes"],
                    width=cfg.get("width", 32), dropout=cfg.get("dropout", 0.3))
