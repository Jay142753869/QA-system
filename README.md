# 金融事件问答系统模块 (Financial Event QA System)

这是一个基于 Flask、Neo4j 和 深度学习(RE-GCN) 技术构建的智能金融事件问答系统。项目创新性地采用了**混合检索策略 (Hybrid Retrieval Strategy)**，结合了基于规则的知识库查询与基于神经网络的时间序列推理，旨在提供既精准又具备预测能力的问答服务。

## 🌟 核心特性

1.  **混合检索与双路输出 (Hybrid Retrieval & Dual Output)**：
    *   **知识库直搜 (Knowledge Base Lookup)**：优先在本地知识库（如 `80stocks_quadruples.csv`）中精确查找历史事实。结果在 UI 中以**绿色**显著标识，代表 100% 的确定性。
    *   **时序推理 (Temporal Reasoning)**：集成 **RE-GCN (Recurrent Event Graph Convolutional Network)** 模型，针对历史缺失或未来时间点进行概率推理。结果在 UI 中以**蓝色**标识，并附带置信度分数（保留两位小数，过滤 < 0.05 的低概率结果）。

2.  **深度 NLP 认知与可视化**：
    *   **动态时间识别**：内置正则引擎，支持识别任意年份（如 `2018年`）、日期（`2024-05`）等时间实体。
    *   **智能实体推断**：结合 AC 自动机（已知实体）与 Jieba POS 词性标注，自动推断未登录词的实体类型。
    *   **认知面板**：在聊天界面实时展示系统的“思考过程”——分词结果、实体类型高亮、意图四元组 `(Head, Relation, Tail, Time)` 解析。

3.  **强大的模型集成**：
    *   内置 RE-GCN 模型，能够处理时序知识图谱（Temporal Knowledge Graph）的补全与预测任务。
    *   解决了“冷启动”问题：针对早期时间点的查询，自动回退到最近的有效历史窗口进行推理。

4.  **多端运行支持**：
    *   **Web 版**：基于 Flask 的 B/S 架构，适合部署在服务器。
    *   **桌面版**：基于 PyWebview 封装的独立 `.exe` 软件，原生窗口体验，无需配置环境即可运行。

## 📂 目录结构

```
问答系统模块/
├── app.py                  # Web 服务入口 (Flask)
├── gui_launcher.py         # 桌面程序入口 (PyWebview)
├── config.py               # 系统配置 (数据库/模型开关)
├── core/                   # 核心算法模块
│   ├── preprocessing.py    # NLP预处理 (AC自动机, 正则时间提取)
│   ├── graph_dao.py        # 数据访问层 (Neo4j/CSV本地查询)
│   ├── reasoning.py        # 推理逻辑封装
│   └── regcn_wrapper.py    # RE-GCN 模型调用封装 (推理/预测)
├── models/                 # 模型与数据仓库
│   └── RE-GCN-master/      # RE-GCN 模型源码与数据
│       ├── src/            # 模型核心代码
│       └── data/           # 训练数据 (部分大文件未上传至GitHub)
├── static/                 # 静态资源 (CSS, JS)
├── templates/              # 前端页面
└── requirements.txt        # Python 依赖列表
```

## 🛠️ 安装与运行

### 环境要求
*   Python 3.8+
*   PyTorch (建议配合 CUDA 使用)
*   DGL (Deep Graph Library)

### 方式一：运行桌面版 (推荐)
直接运行 `dist/` 目录下的 **`FinancialQA_Desktop.exe`** 即可启动独立窗口应用。

### 方式二：源码运行 (开发模式)

1.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    *注意：RE-GCN 相关的依赖可能需要单独安装，具体参考 `models/RE-GCN-master/requirement.txt`*

2.  **准备数据**
    确保 `models/RE-GCN-master/data/` 目录下包含必要的数据集（如 `80STOCKS`, `STRUCTURE` 等）。
    *(注：GitHub 仓库中部分超过 100MB 的训练数据文件已被排除)*

3.  **启动 Web 服务**
    ```bash
    python app.py
    ```
    访问: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

4.  **启动桌面模式**
    ```bash
    python gui_launcher.py
    ```

## 🧪 测试用例

在输入框中尝试以下问题，体验混合检索能力：

*   **知识库精确查询**（绿色结果）：
    *   `招商银行的董事长是谁？`
    *   `平安银行2022年的高管有哪些？`

*   **模型推理预测**（蓝色结果）：
    *   `预测招商银行明年的战略委员会成员`
    *   *(当查询时间超出知识库范围，或知识库缺失时，模型将给出概率预测)*

## ⚙️ 配置说明 (config.py)

*   **`USE_MOCK_MODELS`**:
    *   `True`: 使用模拟/本地 CSV 数据模式（推荐，无需安装 Neo4j）。
    *   `False`: 尝试连接真实 Neo4j 数据库。
*   **RE-GCN 配置**:
    *   模型路径和参数在 `core/regcn_wrapper.py` 中定义。

## 📝 开发进度

- [x] 基础 Web 框架搭建 (Flask)
- [x] NLP 预处理模块 (AC自动机 + 正则 + Jieba)
- [x] 桌面应用程序封装 (.exe)
- [x] **集成 RE-GCN 时序推理模型**
- [x] **实现混合检索策略 (CSV直搜 + 模型推理)**
- [x] **UI 升级：双路结果展示与置信度过滤**
- [ ] 接入实时财经新闻流
