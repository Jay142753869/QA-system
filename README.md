# 金融事件问答系统模块 (Financial Event QA System)

这是一个基于 Flask、Neo4j 和 NLP 技术构建的智能金融事件问答系统。项目集成了知识图谱查询（内推）和事件未来预测（外推）双引擎，并具备强大的**NLP认知可视化**能力。

![系统界面](static/img/preview.png) *(请自行添加截图)*

## 🌟 核心特性

1.  **双引擎驱动**：
    *   **内推引擎 (Internal Reasoning)**：基于知识图谱进行实体关系查询与补全（例如：“贵州茅台2024年的大股东是谁？”）。
    *   **外推引擎 (External Reasoning)**：基于事件演化模型预测未来趋势（例如：“预测某公司股价波动风险”）。

2.  **深度NLP认知与可视化**：
    *   **动态时间识别**：内置正则引擎，支持识别任意年份（如 `2018年`）、日期（`2024-05`）等时间实体。
    *   **智能实体推断**：结合 AC自动机（已知实体）与 Jieba POS 词性标注（未知实体），自动推断未登录词的实体类型。
    *   **认知面板**：在聊天界面实时展示系统的“思考过程”——分词结果、实体类型高亮、意图四元组 `(Head, Relation, Tail, Time)` 解析。

3.  **多端运行支持**：
    *   **Web 版**：基于 Flask 的 B/S 架构，适合部署在服务器。
    *   **桌面版**：基于 PyWebview 封装的独立 `.exe` 软件，原生窗口体验，无需配置环境即可运行。

4.  **无缝 Mock/Real 切换**：
    *   内置 **Mock Engine**，在无真实数据情况下即可演示完整业务流程。
    *   一键配置切换至真实 Neo4j 数据库和深度学习模型。

## 📂 目录结构

```
问答系统模块/
├── app.py              # Web 服务入口 (Flask)
├── gui_launcher.py     # 桌面程序入口 (PyWebview)
├── config.py           # 系统配置 (数据库/模型开关)
├── core/               # 核心算法模块
│   ├── preprocessing.py # NLP预处理 (AC自动机, 正则时间提取, POS推断)
│   ├── graph_dao.py     # 图数据库接口 (Neo4j/Mock)
│   └── reasoning.py     # 推理引擎 (Link Prediction/Event Prediction)
├── static/             # 静态资源 (CSS, JS)
├── templates/          # 前端页面
├── dist/               # 打包后的可执行文件 (.exe)
└── requirements.txt    # Python 依赖列表
```

## 🛠️ 安装与运行

### 方式一：运行桌面版 (推荐)
直接运行 `dist/` 目录下的 **`FinancialQA_Desktop.exe`** 即可启动独立窗口应用。

### 方式二：源码运行 (开发模式)

1.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

2.  **启动 Web 服务**
    ```bash
    python app.py
    ```
    访问: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

3.  **启动桌面模式**
    ```bash
    python gui_launcher.py
    ```

## 🧪 测试用例

在输入框中尝试以下问题，体验系统的认知能力：

*   **基础查询**：`贵州茅台的大股东是谁？`
*   **动态时间**：`贵州茅台2018年的净利润` （系统能识别库中不存在的年份）
*   **未知实体**：`特斯拉2030年的销量` （系统能推断“特斯拉”为实体）

## ⚙️ 配置说明 (config.py)

*   **`USE_MOCK_MODELS`**:
    *   `True` (默认): 使用内置模拟数据。
    *   `False`: 连接真实 Neo4j 并加载本地模型文件。
*   **Neo4j 配置**:
    *   修改 `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`。

## 📦 打包指南

如果需要重新打包 .exe 文件：

```bash
# 打包桌面版
pyinstaller --clean --noconfirm FinancialQA_Desktop.spec
```

## 📝 开发进度

- [x] 基础 Web 框架搭建 (Flask)
- [x] 前端交互界面与可视化 (Bootstrap + jQuery)
- [x] NLP 预处理模块 (AC自动机 + 正则 + Jieba)
- [x] 认知可视化面板 (四元组解析展示)
- [x] 桌面应用程序封装 (.exe)
- [ ] 接入真实 Neo4j 数据库
- [ ] 训练并部署 TAMG 事件预测模型
