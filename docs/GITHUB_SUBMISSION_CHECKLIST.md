# GitHub 提交检查清单

提交 GitHub 前建议逐项确认：

- [ ] `README.md` 中项目说明、运行方式和目录结构准确。
- [ ] `requirements.txt` 与代码实际依赖一致。
- [ ] `src/` 中包含数据下载、预处理、训练评估三类脚本。
- [ ] `docs/DATA.md` 说明数据来源、处理规则和划分方式。
- [ ] `docs/REPRODUCIBILITY.md` 说明完整复现实验步骤。
- [ ] `results/summary.csv` 与报告中的结果表一致。
- [ ] 本地 `report/main.tex`、`report/references.bib`、`report/figures/` 能在 Overleaf 编译。
- [ ] 本地 `report/main.pdf` 已由最新 `main.tex` 编译生成。
- [ ] 报告首页姓名、学号已填写。
- [ ] 作者贡献中的姓名、研究领域已填写。
- [ ] 报告末尾 GitHub 链接已替换为真实仓库地址。
- [ ] 仓库中没有提交 `.venv/`、`__pycache__/`、`.DS_Store`、`data/raw/`、`report/` 和原始数据压缩包。

推荐提交内容：

```text
README.md
requirements.txt
.gitignore
.gitattributes
src/
docs/
data/processed/
results/
figures/
```

不推荐提交内容：

```text
.venv/
data/raw/
report/
src/__pycache__/
.DS_Store
```
