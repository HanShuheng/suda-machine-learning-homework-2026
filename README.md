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
└── docs/                     # 数据与复现说明
```

说明：`data/raw/` 中的原始下载文件不提交到 GitHub，可通过脚本重新下载。

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
python src/run_experiments.py --epochs 50 --seeds 2026 2027 2028 2029 2030
```

快速检查流程是否可运行：

```bash
python src/run_experiments.py --epochs 2 --seeds 2026 --horizons 90 --models lstm transformer hybrid
```

更详细的复现实验说明见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 实验结果

当前实验使用 50 epoch 设置下的可复现实验结果。汇总结果保存在 [results/summary.csv](results/summary.csv)。

| 模型 | 预测长度 | MSE 均值 | MSE 标准差 | MAE 均值 | MAE 标准差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 90 | 481605.93 | 37675.74 | 553.46 | 21.03 |
| Transformer | 90 | 577249.38 | 33514.96 | 600.08 | 18.44 |
| 改进模型 | 90 | 517722.93 | 27129.64 | 566.55 | 13.77 |
| LSTM | 365 | 214145.58 | 23019.90 | 345.55 | 21.33 |
| Transformer | 365 | 335376.71 | 31235.76 | 445.37 | 22.27 |
| 改进模型 | 365 | 220861.83 | 14382.55 | 353.82 | 13.04 |

在当前设置下，LSTM 取得最低平均误差；卷积-金字塔-Transformer 改进模型相较标准 Transformer 在长期预测中明显降低误差和波动，但没有超过 LSTM。

## 代码入口

- `src/download_data.py`：下载课程指定的两个公开数据源。
- `src/preprocess.py`：完成缺失处理、日级聚合、天气合并和 train/test 划分。
- `src/run_experiments.py`：定义 LSTM、Transformer、改进模型，完成训练、评估、保存指标和绘图。
