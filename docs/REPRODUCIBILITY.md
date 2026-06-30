# 复现实验说明

本文实验基于 Python、PyTorch、pandas 和 scikit-learn。建议在独立虚拟环境中运行。

## 环境准备

```bash
cd /Users/hanshuheng/codes/CodeX/机器学习大作业/大作业
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需在其他机器运行，请先进入仓库根目录，再执行同样命令。

## 完整复现流程

```bash
python src/download_data.py
python src/preprocess.py
python src/run_experiments.py --epochs 50 --seeds 2026 2027 2028 2029 2030
```

输出文件：

```text
data/processed/train.csv
data/processed/test.csv
results/detail.csv
results/summary.csv
results/runs.json
figures/
```

结果表使用 `results/summary.csv` 汇总，预测曲线使用 `figures/` 中对应图片。

## 快速冒烟测试

完整训练耗时较长。只检查代码是否能跑通时，可以执行：

```bash
python src/run_experiments.py --epochs 2 --seeds 2026 --horizons 90 --models lstm transformer hybrid
```

该命令只用于验证流程，不应用作最终实验结果。

## 当前实验设置

- 输入窗口：过去 90 天。
- 预测长度：未来 90 天和未来 365 天，分别训练模型。
- 模型：LSTM、Transformer、卷积-金字塔-Transformer 混合模型。
- 随机种子：2026、2027、2028、2029、2030。
- 指标：MSE、MAE，统计 5 轮实验均值和标准差。
