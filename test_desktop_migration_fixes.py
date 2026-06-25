#!/usr/bin/env python
"""
回归验证脚本 - 验证第二批桌面迁移与推理稳定性修复。

覆盖项:
  桌面端自动端口与关闭清理
  CDN 本地化与 vendor 资源清单
  文件日志位置与轮转
  S4 推理 predict 独立锁
  H2 NLP 关系回退误判收紧
  train_model.py 使用环境变量配置解释器
  README/使用指南关键说明同步

用法:
  python test_desktop_migration_fixes.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def read_text(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        return f.read()


def test_desktop_launcher():
    print("\n=== 桌面端自动端口与清理 ===")
    content = read_text("gui_launcher.py")
    check("使用 werkzeug.make_server", "make_server" in content)
    check("绑定 127.0.0.1 且 port=0", 'make_server("127.0.0.1", 0, app)' in content)
    check("读取实际分配端口", "_get_server_port()" in content and "_server.server_port" in content)
    check("窗口关闭时清理服务", "window.events.closed += shutdown_server" in content)
    check("清理 GraphDAO 连接", "state.graph_dao.close()" in content)


def test_vendor_localization():
    print("\n=== CDN 本地化 ===")
    template = read_text("templates/index.html")
    check("模板不再引用 bootcdn", "cdn.bootcdn.net" not in template)
    check("Bootstrap CSS 使用本地 vendor", "vendor/css/bootstrap.min.css" in template)
    check("FontAwesome CSS 使用本地 vendor", "vendor/css/all.min.css" in template)
    check("Bootstrap JS 使用本地 vendor", "vendor/js/bootstrap.bundle.min.js" in template)
    check("jQuery 使用本地 vendor", "vendor/js/jquery.min.js" in template)

    required = [
        "static/vendor/css/bootstrap.min.css",
        "static/vendor/css/all.min.css",
        "static/vendor/js/bootstrap.bundle.min.js",
        "static/vendor/js/jquery.min.js",
        "static/vendor/webfonts/fa-solid-900.woff2",
        "static/vendor/SOURCES.md",
    ]
    for rel_path in required:
        check(f"存在 {rel_path}", os.path.exists(os.path.join(ROOT, rel_path)))


def test_logging_and_training_config():
    print("\n=== 文件日志与训练路径配置 ===")
    app_content = read_text("app.py")
    train_content = read_text("train_model.py")
    check("app.py 使用 RotatingFileHandler", "RotatingFileHandler" in app_content)
    check("日志写入 LOCALAPPDATA/FinancialQA/logs", "LOCALAPPDATA" in app_content and "FinancialQA" in app_content and "logs" in app_content)
    check("logging.basicConfig 使用 force=True 保证文件 handler 生效", "force=True" in app_content)
    check("train_model.py 从环境变量读取解释器", 'os.environ.get("ANACONDA_PYTHON_PATH")' in train_content)
    check("train_model.py 不再硬编码 Jay14 路径", "C:\\Users\\Jay14\\anaconda3" not in train_content)


def test_reasoning_predict_locks():
    print("\n=== S4 推理线程安全 ===")
    content = read_text("core/reasoning.py")
    check("REGCN predict 独立锁存在", "_regcn_predict_lock = threading.Lock()" in content)
    check("TiRGN predict 独立锁存在", "_tirgn_predict_lock = threading.Lock()" in content)
    check("REGCN predict 调用被锁保护", "with self._regcn_predict_lock:" in content and "model.predict(" in content)
    check("TiRGN predict 调用被锁保护", "with self._tirgn_predict_lock:" in content and "model.predict_tail(" in content)


class FakeACMatcher:
    def __init__(self, entity):
        self.entity = entity

    def search(self, text):
        start = text.find(self.entity)
        if start < 0:
            return []
        return [{
            "word": self.entity,
            "start": start,
            "end": start + len(self.entity) - 1,
            "type": "AC_MATCH",
        }]


class FakeMultiACMatcher:
    def __init__(self, entities):
        self.entities = entities

    def search(self, text):
        matches = []
        for entity in self.entities:
            start = text.find(entity)
            if start < 0:
                continue
            matches.append({
                "word": entity,
                "start": start,
                "end": start + len(entity) - 1,
                "type": "AC_MATCH",
            })
        return matches


class FakeSimilarity:
    def __init__(self, scores):
        self.scores = scores

    def encode(self, text):
        return np.ones(3)

    def compute_similarity(self, query_emb, candidate_embs):
        scores = np.array(self.scores, dtype=float)
        return np.arange(len(scores)), scores


def make_processor(scores):
    from core.preprocessing import NLPProcessor

    processor = NLPProcessor.__new__(NLPProcessor)
    processor.config = {}
    processor.ac_matcher = FakeACMatcher("招商银行")
    processor.sim_model = FakeSimilarity(scores)
    processor.relation_list = ["董事会秘书", "联席总裁"]
    processor.relation_embs = np.ones((2, 3))
    processor._relation_embs_ready = True
    processor._relation_embs_lock = None
    processor._ensure_relation_embeddings = lambda: None
    return processor


def test_h2_relation_fallback():
    print("\n=== H2 NLP 关系回退误判 ===")

    blacklist_processor = make_processor([0.99, 0.70])
    blacklist_result = blacklist_processor.analyze("招商银行的首席厨师是谁？")
    check(
        "黑名单词不被强行映射为关系",
        blacklist_result["structured_query"]["r"] == "",
        blacklist_result["structured_query"],
    )

    margin_processor = make_processor([0.96, 0.91])
    margin_result = margin_processor.analyze("招商银行的职责是谁？")
    check(
        "Top1/Top2 margin 不足时拒绝关系回退",
        margin_result["structured_query"]["r"] == "",
        margin_result["structured_query"],
    )

    valid_processor = make_processor([0.96, 0.70])
    valid_result = valid_processor.analyze("招商银行的董事会秘书是谁？")
    check(
        "高置信且高 margin 时仍可回退识别关系",
        valid_result["structured_query"]["r"] == "董事会秘书",
        valid_result["structured_query"],
    )


def test_segmentation_preserves_ac_entities():
    print("\n=== NLP 展示分词保留已识别实体 ===")
    from core.preprocessing import NLPProcessor

    processor = NLPProcessor.__new__(NLPProcessor)
    processor.config = {}
    processor.ac_matcher = FakeACMatcher("蔡洪平")
    processor.sim_model = FakeSimilarity([0.0])
    processor.relation_list = []
    processor.relation_embs = None
    processor._relation_embs_ready = True
    processor._relation_embs_lock = None
    processor._ensure_relation_embeddings = lambda: None

    result = processor.analyze("蔡洪平和招商银行之间是什么关系?")
    tokens = [item["word"] for item in result["segmentation"]]

    check(
        "展示分词包含完整实体蔡洪平",
        "蔡洪平" in tokens,
        f"tokens={tokens}",
    )
    check(
        "展示分词不再把蔡洪平切成蔡洪/平和",
        "蔡洪" not in tokens and "平和" not in tokens,
        f"tokens={tokens}",
    )


def test_structured_query_preserves_second_entity():
    print("\n=== NLP 四元组保留第二实体 ===")
    from core.preprocessing import NLPProcessor

    processor = NLPProcessor.__new__(NLPProcessor)
    processor.config = {}
    processor.ac_matcher = FakeMultiACMatcher(["蔡洪平", "招商银行"])
    processor.sim_model = FakeSimilarity([0.0])
    processor.relation_list = []
    processor.relation_embs = None
    processor._relation_embs_ready = True
    processor._relation_embs_lock = None
    processor._ensure_relation_embeddings = lambda: None

    result = processor.analyze("2025年1月，蔡洪平在招商银行一直担任什么职务?")
    structured = result["structured_query"]

    check(
        "第一个实体进入 h",
        structured["h"] == "蔡洪平",
        structured,
    )
    check(
        "第二个实体进入 t",
        structured["t"] == "招商银行",
        structured,
    )
    check(
        "月份时间仍可识别",
        structured["time"] == "2025年1月",
        structured,
    )


def test_docs_synced():
    print("\n=== 文档同步 ===")
    readme = read_text("README.md")
    guide = read_text("使用指南.md")
    checklist = read_text("交付清单.md")
    check("README 说明桌面启动", "python gui_launcher.py" in readme)
    check("README 说明日志位置", "%LOCALAPPDATA%\\FinancialQA\\logs\\app.log" in readme)
    check("README 说明 USE_DEMO_MOCK_GRAPH", "USE_DEMO_MOCK_GRAPH" in readme)
    check("使用指南说明自动端口/本地vendor", "自动空闲端口" in guide and "本地 vendor" in guide)
    check("交付清单说明目标机打包验收", "目标机器" in checklist)


if __name__ == "__main__":
    print("=" * 60)
    print("回归验证 - 第二批桌面迁移与稳定性修复")
    print("=" * 60)

    test_desktop_launcher()
    test_vendor_localization()
    test_logging_and_training_config()
    test_reasoning_predict_locks()
    test_h2_relation_fallback()
    test_segmentation_preserves_ac_entities()
    test_structured_query_preserves_second_entity()
    test_docs_synced()

    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
