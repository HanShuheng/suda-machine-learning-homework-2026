# 数据说明

本项目使用课程考核 PDF 指定的两个公开数据源。仓库中不提交原始压缩包，原始数据通过脚本下载，处理后的日级数据可由脚本重新生成。

## 原始数据来源

- UCI 家庭电力分钟级数据：Individual household electric power consumption  
  https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Meteo-France 月度基础气候数据：Donnees climatologiques de base - mensuelles  
  https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles

默认下载法国 92 省 Hauts-de-Seine 的月度气象数据，用于近似匹配 UCI 数据中法国巴黎近郊家庭的地理位置。

## 数据处理规则

电力数据由分钟级聚合到日级：

- `Global_active_power`、`Global_reactive_power`、`Sub_metering_1`、`Sub_metering_2`、`Sub_metering_3`：按天求和。
- `Voltage`、`Global_intensity`：按天求平均。
- `Sub_metering_remainder`：按课程提示由总有功功率与三个分表计算。
- 缺失日期：按日频补齐，并对数值字段进行线性插值。

气象数据为月度粒度，实验中将同月气象值合并到该月每一天：

- `RR`
- `NBJRR1`
- `NBJRR5`
- `NBJRR10`
- `NBJBROU`

## 训练与测试划分

- 训练集：2006-12-16 至 2008-12-31，共 747 天。
- 测试集：2009-01-01 至 2010-11-26，共 695 天。

模型训练时只使用训练集拟合归一化器，再将同一变换应用到测试集，避免测试数据泄漏。

## 重新生成数据

```bash
python src/download_data.py
python src/preprocess.py
```

生成结果：

```text
data/processed/train.csv
data/processed/test.csv
```
