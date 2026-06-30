# 家庭电力消耗多变量时间序列预测

本仓库为 2026 年专硕机器学习课程项目，任务是基于家庭电力消耗数据和月度气象数据，使用过去 90 天的多变量日级序列预测未来 90 天和 365 天的每日总有功功率。

项目实现了三类模型：

- LSTM 基线模型
- Transformer 基线模型
- 卷积-金字塔-Transformer 改进模型

评价指标为 MSE 和 MAE。每个模型在 90 天与 365 天两个预测任务上分别训练，并使用 5 个随机种子统计均值和标准差。

## 数据来源

本项目使用课程考核要求中的公开数据源：

- UCI Machine Learning Repository: Individual household electric power consumption  
  https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- data.gouv.fr / Meteo-France: Donnees climatologiques de base - mensuelles  
  https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles

数据处理细节见 [docs/DATA.md](docs/DATA.md)。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── download_data.py      # 下载 UCI 电力数据和 Meteo-France 气象数据
│   ├── preprocess.py         # 分钟级数据聚合为日级 train/test
│   └── run_experiments.py    # 训练、评估、绘图
├── data/
│   └── processed/            # 已处理的日级 train.csv / test.csv
├── results/                  # 实验指标明细与汇总
├── figures/                  # 训练损失图与预测曲线
└── docs/                     # 数据、复现和提交说明
```

说明：`data/raw/` 中的原始下载文件不建议提交到 GitHub，可通过脚本重新下载；`report/` 为本地报告与 Overleaf 文件目录，不提交到公开仓库。

## 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖：

- numpy
- pandas
- scikit-learn
- matplotlib
- torch

## 运行流程

下载原始数据：

```bash
python src/download_data.py
```

生成日级训练集和测试集：

```bash
python src/preprocess.py
```

运行完整实验：

```bash
python src/run_experiments.py --epochs 80 --seeds 2026 2027 2028 2029 2030
```

快速检查流程是否可运行：

```bash
python src/run_experiments.py --epochs 2 --seeds 2026 --horizons 90 --models lstm transformer hybrid
```

更详细的复现实验说明见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 当前实验结果

当前报告使用 10 epoch 设置下的可复现实验结果。汇总结果保存在 [results/summary.csv](results/summary.csv)。

| 模型 | 预测长度 | MSE 均值 | MSE 标准差 | MAE 均值 | MAE 标准差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 90 | 463333.44 | 22846.57 | 540.50 | 14.37 |
| Transformer | 90 | 530704.40 | 52576.14 | 578.32 | 35.62 |
| 改进模型 | 90 | 488765.94 | 48946.29 | 553.60 | 30.14 |
| LSTM | 365 | 249855.29 | 4756.52 | 387.51 | 4.13 |
| Transformer | 365 | 277300.21 | 5238.13 | 408.81 | 4.53 |
| 改进模型 | 365 | 255940.64 | 1143.36 | 392.37 | 1.57 |

在当前设置下，LSTM 取得最低平均误差；卷积-金字塔-Transformer 改进模型相较标准 Transformer 在长期预测中明显降低误差和波动，但没有超过 LSTM。

## 报告与 Overleaf

报告源码、参考文献、PDF 和 Overleaf 上传文件位于本地 `report/` 目录。为避免公开仓库中出现可被直接复用的文章正文，该目录已加入 `.gitignore`，不会提交到 GitHub。

本地 Overleaf 上传内容：

```text
report/main.tex
report/references.bib
report/figures/
```

Overleaf 设置：

- Compiler: XeLaTeX
- Main document: `main.tex`

上传说明保存在本地 `report/README_OVERLEAF.md`。

## 提交前需要填写

报告中仍需按个人信息手动填写：

- 首页姓名、学号
- 作者贡献中的姓名和研究领域
- 文末实际 GitHub 仓库链接

提交 GitHub 前建议检查 [docs/GITHUB_SUBMISSION_CHECKLIST.md](docs/GITHUB_SUBMISSION_CHECKLIST.md)。

## 代码入口

- `src/download_data.py`：下载课程指定的两个公开数据源。
- `src/preprocess.py`：完成缺失处理、日级聚合、天气合并和 train/test 划分。
- `src/run_experiments.py`：定义 LSTM、Transformer、改进模型，完成训练、评估、保存指标和绘图。

## 说明

本项目报告文字整理、LaTeX 排版和部分表述润色过程中使用了 ChatGPT/Codex 作为写作辅助工具；数据处理、模型训练、实验指标和预测曲线均由项目代码生成，并在本地环境完成验证。所有外部数据来源和参考文献已在报告中列出。
