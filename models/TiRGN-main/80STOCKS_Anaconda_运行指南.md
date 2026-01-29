# 80STOCKS（本地数据集）在 Anaconda 下运行 TiRGN 操作指南

本项目里的 “80STOCKS 新数据库” 并不是 MySQL/PostgreSQL/SQLite 这类数据库服务，而是一个**本地文件型的时间知识图谱数据集（Temporal KG）**，目录位于 [data/80STOCKS](file:///d:/研究项目/TiRGN/TiRGN-main/data/80STOCKS)。

TiRGN 的运行流程可以概括为：
1. 准备/检查 `data/80STOCKS/` 下的数据文件是否齐全。
2. 生成历史缓存（history）文件：`src/get_history.py` 会写出 `data/80STOCKS/history/*.npz`。
3. 训练/测试：`src/main.py` 读取数据 + history，进行训练或评测。

---

## 1. 环境准备（Anaconda）

### 1.1 创建并激活环境

项目 README 给出的 Python 版本是 3.7（建议保持一致）：

```bash
conda create -n tirgn python=3.7
conda activate tirgn
```

### 1.2 安装依赖

仓库依赖文件为 [requirement.txt](file:///d:/研究项目/TiRGN/TiRGN-main/requirement.txt)。其中包含 `torch==1.6.0`、`dgl-cu102==0.5.2`（CUDA 10.2 对应的 DGL 轮子）等固定版本。

#### 方案 A：机器有 CUDA 10.2（与 requirements 完全一致）

```bash
pip install -r requirement.txt
```

#### 方案 B：不使用 GPU（CPU 运行）

如果你的机器没有 CUDA 10.2（或想在 CPU 上跑通流程），通常需要：
1. 安装 **CPU 版** PyTorch（建议用 conda 官方渠道更稳）。
2. 把 `dgl-cu102==0.5.2` 改为 CPU 版 DGL（例如 `dgl==0.5.2`），再安装其它依赖。

说明：
- 本仓库的 `requirement.txt` 固定写了 `dgl-cu102==0.5.2`，这会在无 CUDA10.2 的环境下导致安装/导入失败。
- DGL/PyTorch 的 wheel/conda 包在不同系统与 CUDA 版本下差异很大，按你本机 CUDA/驱动情况选择对应安装方式即可。

你最终应确保下面这些 import 能成功（任意 Python 解释器里执行）：
```python
import torch
import dgl
import numpy as np
import scipy
```

---

## 2. 数据准备与目录检查（80STOCKS）

80STOCKS 数据集目录：[data/80STOCKS](file:///d:/研究项目/TiRGN/TiRGN-main/data/80STOCKS)

### 2.1 必需文件（训练/测试必用）

TiRGN 的本地加载会读取：
- `entity2id.txt`
- `relation2id.txt`
- `train.txt`
- `valid.txt`
- `test.txt`

对应加载入口在 [knowledge_graph.py](file:///d:/研究项目/TiRGN/TiRGN-main/rgcn/knowledge_graph.py#L173-L213)。

### 2.2 生成 history 必需文件

[src/get_history.py](file:///d:/研究项目/TiRGN/TiRGN-main/src/get_history.py#L86-L123) 还会读取：
- `stat.txt`（前两列分别是实体数、关系数；例如 80STOCKS 为 `2581  237  380`）
- `train.txt / valid.txt / test.txt`

### 2.3 静态图文件（可选）

若训练时加 `--add-static-graph`，还需要：
- `e-w-graph.txt`（80STOCKS 目录内已提供）

读取位置在 [main.py](file:///d:/研究项目/TiRGN/TiRGN-main/src/main.py#L183-L190)。

---

## 3. 代码准备：放行 80STOCKS 数据集名称

TiRGN 通过 `utils.load_data()` 对可用数据集做了白名单判断。为了让 `-d 80STOCKS` 走本地目录加载，需要在白名单中包含 `80STOCKS`。

本仓库已在 [utils.py](file:///d:/研究项目/TiRGN/TiRGN-main/rgcn/utils.py#L323-L333) 中加入 `"80STOCKS"`，如果你本地代码仍报 `Unknown dataset: 80STOCKS`，请确认该处修改存在。

---

## 4. 生成 history（必须做一次）

在仓库根目录执行：

```bash
cd src
python get_history.py --dataset 80STOCKS
```

成功后会生成目录：
- `data/80STOCKS/history/`
并产出：
- `tail_history_*.npz`
- `rel_history_*.npz`

---

## 5. 训练 80STOCKS

`src/main.py` 的 `-d/--dataset` 是必填参数，其它都有默认值。建议先用 CPU 跑通流程（`--gpu -1` 是默认值）。

在 `src/` 目录运行一个参考命令（参数风格沿用 README 示例）：

```bash
python main.py -d 80STOCKS --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5 --discount 1 --task-weight 0.7 --entity-prediction --relation-prediction --add-static-graph --gpu -1 --save checkpoint
```

说明：
- 如果你的 `data/80STOCKS/` 没有 `e-w-graph.txt`，请去掉 `--add-static-graph`。
- 训练过程中模型会保存到 `models/` 目录（相对仓库根目录的 `../models/`，在 [main.py](file:///d:/研究项目/TiRGN/TiRGN-main/src/main.py#L174-L179) 里拼出来）。

---

## 6. 测试/评估

评测用训练同样的参数，加上 `--test`：

```bash
python main.py -d 80STOCKS --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5 --discount 1 --task-weight 0.7 --entity-prediction --relation-prediction --add-static-graph --gpu -1 --save checkpoint --test
```

`--test` 模式会加载 `models/` 下同名 checkpoint 并开始测试，加载逻辑见 [main.py](file:///d:/研究项目/TiRGN/TiRGN-main/src/main.py#L29-L38)。

---

## 7. 常见问题排查

### 7.1 `ValueError: Unknown dataset: 80STOCKS`

确认 [utils.load_data](file:///d:/研究项目/TiRGN/TiRGN-main/rgcn/utils.py#L323-L333) 白名单包含 `80STOCKS`。

### 7.2 `ModuleNotFoundError: No module named 'scipy'`

history 生成和训练都会用到 SciPy（见 [get_history.py](file:///d:/研究项目/TiRGN/TiRGN-main/src/get_history.py#L4-L6) 与 [main.py](file:///d:/研究项目/TiRGN/TiRGN-main/src/main.py#L21-L22)）。请在 conda 环境中安装：
```bash
pip install scipy
```
或
```bash
conda install scipy
```

### 7.3 DGL / CUDA 相关报错

如果安装的是 `dgl-cu102==0.5.2`，运行时需要机器满足 CUDA 10.2 相关条件；不满足时建议切换到 CPU 版 DGL + CPU 版 PyTorch，再用 `--gpu -1` 跑通流程。

---

## 8. 如果你要“自制一个新的 80STOCKS 数据库”（从 CSV 开始）

本仓库内存在原始 CSV：[80stocks_quadruples.csv](file:///d:/研究项目/TiRGN/TiRGN-main/data/80STOCKS/80stocks_quadruples.csv)，但仓库**没有**提供从 CSV 自动生成 `entity2id/relation2id/time2id + train/valid/test` 的脚本。

要让 TiRGN 能读取一个新的本地数据集目录（例如 `data/80STOCKS_NEW/`），至少需要生成以下文件：
- `entity2id.txt`：实体字符串到整数 ID 的映射
- `relation2id.txt`：关系字符串到整数 ID 的映射
- `time2id.txt`：时间字符串/时间戳到整数 ID 的映射（可选，但建议保留）
- `train.txt / valid.txt / test.txt`：每行四元组 `h_id r_id t_id time_id_or_time`（空格分隔）
- `stat.txt`：至少包含两列：`num_entities  num_relations`（第三列可以是时间数）

并且需要在 [utils.load_data](file:///d:/研究项目/TiRGN/TiRGN-main/rgcn/utils.py#L323-L333) 中把你的新数据集名字（如 `80STOCKS_NEW`）加入白名单，之后重复本指南的第 4～6 节即可。

