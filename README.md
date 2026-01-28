# 金融事件问答系统模块 (Financial Event QA System)

这是一个基于 Flask 的金融问答系统。系统对用户问题进行 NLP 结构化解析后，同时给出两类结果：

1. **知识库查询结果**：从 Neo4j 或本地 CSV 知识库中检索已存在的事实；
2. **模型推理结果**：使用 RE-GCN 模型对缺失事实进行补全（以概率形式给出候选答案）。

## 核心能力

### 1) 双路输出（知识库 + 模型推理）

- **知识库查询（第一行）**：优先走 Neo4j；当未启用 Neo4j 或连接失败时，使用本地 CSV（默认：`models/RE-GCN-master/data/80STOCKS/80stocks_quadruples.csv`）进行查询。
  - 当用户只提供“年份/年月”而非精确到“年月日”时，本地 CSV 查询会在该时间前缀范围内选择“最新日期”的记录作为结果（属于“取最新窗口”的策略，不是返回该范围内所有事实）。
- **模型推理（第二行）**：在“内推模式（internal）”下，即使知识库有结果也会继续运行 RE-GCN 推理，返回 Top-K 候选尾实体与概率分数。
  - 概率分数保留两位小数，并过滤掉 `< 0.05` 的结果。

### 2) NLP 结构化解析与可视化

- **实体/关系识别**：通过 AC 自动机从数据集词表中识别实体与关系；无法识别实体时，会从 Jieba 词性中挑选专有名词（`nr/ns/nt/nz`）作为“推断实体（ENTITY (Inferred)）”的回退策略。
- **时间识别**：支持识别 `YYYY`、`YYYY年`、`YYYY-MM`、`YYYYMMDD` 等格式。
- **关系回退匹配**：当未识别到关系时，会将剩余文本与 `relation2id.txt` 的关系词表做 BERT 向量相似度匹配（带阈值）。

### 3) 外推模式说明（external）

当前 `external` 模式仅在 Mock 模式下返回演示用的固定事件描述；非 Mock 模式下暂未接入真实外推模型（会返回空列表）。

## 目录结构

```
问答系统模块/
├── app.py                  # Web 服务入口（Flask + /api/query）
├── gui_launcher.py         # 桌面程序入口（PyWebview）
├── config.py               # 开关与路径配置（Mock/Neo4j/RE-GCN）
├── core/
│   ├── preprocessing.py    # NLP：AC 自动机、时间识别、关系 BERT 匹配
│   ├── graph_dao.py        # 图/本地 CSV 查询（Neo4j 失败则回退）
│   ├── reasoning.py        # 推理引擎（Mock 或 RE-GCN）
│   └── regcn_wrapper.py    # RE-GCN 推理封装
├── models/RE-GCN-master/   # RE-GCN 源码、数据与权重
├── static/                 # 前端静态资源（CSS/JS）
├── templates/              # 页面模板
└── dist/                   # 已打包可执行文件（如已生成）
```

## 运行方式

### 方式一：运行可执行文件（推荐）

仓库内已包含打包产物时，可直接运行：

- `dist/FinancialQA.exe` 或 `dist/FinancialQA/FinancialQA.exe`

### 方式二：源码运行

1) 安装基础依赖（用于 Web + NLP + torch）

```bash
pip install -r requirements.txt
```

2) 桌面模式依赖（仅源码运行桌面版需要）

`gui_launcher.py` 依赖 `pywebview`，如需源码启动桌面版请额外安装：

```bash
pip install pywebview
```

3) 启用 RE-GCN 真模型（可选）

若 `config.py` 中 `USE_MOCK_MODELS=False`，会加载 RE-GCN 进行推理。RE-GCN 需要额外依赖（特别是 DGL），请参考：

- `models/RE-GCN-master/requirement.txt`

4) 启动服务

```bash
python app.py
```

访问：`http://127.0.0.1:5000/`

## 配置说明（config.py）

- `USE_MOCK_GRAPH`
  - `True`：不连接 Neo4j，优先走本地 CSV 查询（并可回退到少量 hardcode mock）。
  - `False`：尝试连接 Neo4j（注意：当前 Cypher 语句要求关系类型为 `RELATION` 且关系属性包含 `name/time` 字段）。
- `USE_MOCK_MODELS`
  - `True`：使用 Mock 推理结果（并降低对本地 RE-GCN 环境依赖）。
  - `False`：加载 RE-GCN 模型进行推理（需要模型权重与 RE-GCN 依赖齐全）。

## 测试问题（建议）

以下问题适合验证“知识库第一行 + 模型第二行”的双路输出是否正常：

- `招商银行的董事长是谁？`
- `贵州茅台的大股东是谁？`
- `贵州茅台2018年的独立董事是谁？`
- `平安银行2022年的高管有哪些？`

## GitHub 大文件说明

由于 GitHub 单文件大小限制（100MB），以下训练数据文件不会被推送到仓库（但可以保留在本地运行环境中）：

- `models/RE-GCN-master/data/STRUCTURE/train.txt`
- `models/RE-GCN-master/data/STRUCTURE/valid.txt`
- `models/RE-GCN-master/data/STRUCTURE/test.txt`
