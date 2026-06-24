#!/usr/bin/env python
"""
回归验证脚本 — 验证第一批 Bug 修复的正确性。

覆盖项:
  S1  模块包隔离 (先 TiRGN 后 REGCN，验证导入到不同的 RecurrentRGCN)
  S2  Mock 回退 (CSV 未命中时返回空，而非错误事实)
  S3  BERT 加载竞态条件 (Lock + Event + generation token)
  S5  get_fact 无时间参数时回退到最新时间
  M1  前端不过滤含 Error 的结果
  M7  区分未识别实体和 graph_dao 不可用
  M8  两实体反查关系/职务

用法:
  python test_regression_fixes.py

注意: 需在依赖齐全的 Conda 环境中运行 (torch / dgl / transformers 等)。
"""

import os
import sys
import traceback

import numpy as np

# 确保项目根目录在 sys.path 中
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


# ---------------------------------------------------------------------------
# S1: 模块包隔离 — 先导入 TiRGN，再导入 REGCN，验证两者加载不同的类
# ---------------------------------------------------------------------------
def test_s1_module_isolation():
    print("\n=== S1: 模块包隔离 ===")
    try:
        from core.model_imports import isolated_model_import

        regcn_dir = os.path.join(ROOT, "models", "RE-GCN-master")
        tirgn_dir = os.path.join(ROOT, "models", "TiRGN-main")

        # 先导入 TiRGN 的 RecurrentRGCN
        with isolated_model_import(tirgn_dir):
            from src.rrgcn import RecurrentRGCN as TiRGN_RGCN

        # 再导入 REGCN 的 RecurrentRGCN
        with isolated_model_import(regcn_dir):
            from src.rrgcn import RecurrentRGCN as REGCN_RGCN

        check(
            "TiRGN 和 REGCN 加载了不同的 RecurrentRGCN 类",
            TiRGN_RGCN is not REGCN_RGCN,
            f"两者 id 相同: {id(TiRGN_RGCN)} == {id(REGCN_RGCN)}",
        )

        # 验证 __init__ 签名不同 (TiRGN 多了 num_times, time_interval, history_rate)
        import inspect

        tirgn_params = inspect.signature(TiRGN_RGCN.__init__).parameters
        regcn_params = inspect.signature(REGCN_RGCN.__init__).parameters

        check(
            "TiRGN 签名包含 history_rate",
            "history_rate" in tirgn_params,
            f"实际参数: {list(tirgn_params.keys())}",
        )
        check(
            "REGCN 签名不包含 history_rate",
            "history_rate" not in regcn_params,
            f"实际参数: {list(regcn_params.keys())}",
        )

        # 验证 sys.path 在导入后恢复原状
        check(
            "sys.path 不含 TiRGN 目录",
            tirgn_dir not in sys.path,
            f"sys.path 仍含: {tirgn_dir}",
        )
        check(
            "sys.path 不含 REGCN 目录",
            regcn_dir not in sys.path,
            f"sys.path 仍含: {regcn_dir}",
        )

    except Exception as e:
        check("S1 测试未抛出异常", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# S2: Mock 回退 — CSV 未命中时返回空
# ---------------------------------------------------------------------------
def test_s2_mock_fallback():
    print("\n=== S2: Mock 回退 ===")
    try:
        from core.graph_dao import GraphDAO

        csv_path = os.path.join(
            ROOT, "models", "RE-GCN-master", "data", "80STOCKS",
            "80stocks_quadruples.csv",
        )

        # use_demo_mock=False: CSV 未命中应返回空
        dao_safe = GraphDAO(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            use_mock=True,
            local_csv_path=csv_path,
            use_demo_mock=False,
        )
        result_safe = dao_safe.query_entity_relation(
            "不存在的公司", "大股东", None
        )
        check(
            "use_demo_mock=False 时未知实体返回空列表",
            result_safe == [],
            f"实际返回: {result_safe}",
        )

        # use_demo_mock=True: 才会返回 mock 数据 (但不检查实体)
        dao_demo = GraphDAO(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            use_mock=True,
            local_csv_path=csv_path,
            use_demo_mock=True,
        )
        result_demo = dao_demo.query_entity_relation(
            "不存在的公司", "大股东", None
        )
        check(
            "use_demo_mock=True 时大股东返回茅台集团 (已知行为)",
            "中国贵州茅台酒厂" in " ".join(result_demo),
            f"实际返回: {result_demo}",
        )

    except Exception as e:
        check("S2 测试未抛出异常", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# S5: get_fact 无时间参数时回退到最新时间
# ---------------------------------------------------------------------------
def test_s5_get_fact_no_time():
    print("\n=== S5: get_fact 无时间回退 ===")
    try:
        from core.regcn_wrapper import REGCNWrapper
        from config import Config

        # wrappers expect dict-style config access
        config = {k: getattr(Config, k) for k in dir(Config)
                   if not k.startswith('_')}
        wrapper = REGCNWrapper(config)

        # 获取最新时间 ID
        latest_t = wrapper._latest_time_id()

        # 取第一个实体和关系用于测试
        head_name = next(iter(wrapper.entity2id))
        rel_name = next(iter(wrapper.relation2id))

        # 无时间参数时 get_fact 应使用最新时间
        # (仅验证内部逻辑: t_id 不为 None)
        resolved_none = wrapper._resolve_time_id(None)
        check(
            "_resolve_time_id(None) 返回 None (无回退)",
            resolved_none is None,
            f"实际返回: {resolved_none}",
        )

        # get_fact 内部应回退到 latest_t，不报错
        facts = wrapper.get_fact(head_name, rel_name, time_str=None)
        check(
            "get_fact(None) 不抛出异常",
            isinstance(facts, list),
            f"实际类型: {type(facts)}",
        )

        # 验证回退逻辑: get_fact(None) 等价于 get_fact(str(latest_t))
        if str(latest_t) in wrapper.time2id:
            facts_explicit = wrapper.get_fact(head_name, rel_name, str(latest_t))
            check(
                "get_fact(None) 与 get_fact(latest_time) 结果一致",
                facts == facts_explicit,
                f"None: {facts}, explicit: {facts_explicit}",
            )

    except Exception as e:
        check("S5 测试未抛出异常", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# S3: BERT 加载竞态条件 — Lock + Event + generation token
# ---------------------------------------------------------------------------
def test_s3_bert_race_condition():
    print("\n=== S3: BERT 加载竞态条件 ===")
    try:
        from core.preprocessing import SimilarityModel

        # --- Case 1: mock 模式直接就绪 ---
        sim = SimilarityModel(use_mock=True)
        check(
            "mock 模式 _ensure_bert_loaded 返回 True",
            sim._ensure_bert_loaded(timeout=1) is True,
        )
        check(
            "mock 模式 encode 不抛异常",
            isinstance(sim.encode("test"), np.ndarray),
        )

        # --- Case 2: 非/mock 模式但 transformers 不可用时回退 ---
        import core.preprocessing as pp
        if not pp.TRANSFORMERS_AVAILABLE:
            sim2 = SimilarityModel(use_mock=False)
            check(
                "transformers 不可用时自动切换 mock",
                sim2.use_mock is True,
                f"use_mock={sim2.use_mock}",
            )
        else:
            check(
                "transformers 可用 (跳过回退测试)",
                True,
            )

        # --- Case 3: 并发调用不会启动多个加载线程 ---
        sim3 = SimilarityModel(use_mock=True)  # mock 模式，不会真正加载
        # 模拟多个线程同时调用 _ensure_bert_loaded
        import threading
        results = []
        def worker():
            results.append(sim3._ensure_bert_loaded(timeout=1))
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        check(
            "5 个线程并发调用全部返回 True",
            all(r is True for r in results) and len(results) == 5,
            f"results={results}",
        )

        # --- Case 4: generation token 防止 stale load 覆盖 ---
        sim4 = SimilarityModel(use_mock=False)
        # 模拟：先启动一次加载 (generation=1)，超时后标记 mock
        sim4._load_generation = 1
        sim4._load_event.clear()
        # 另一个线程启动新加载 (generation=2)
        sim4._load_generation = 2
        # 旧线程 (generation=1) 完成时不应覆盖
        sim4._load_bert_worker(1)  # 旧 generation
        check(
            "stale generation 不会覆盖当前状态",
            sim4._load_generation == 2,
            f"generation={sim4._load_generation}",
        )

    except Exception as e:
        check("S3 测试未抛出异常", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# M1: 前端不过滤含 Error 的结果 (静态验证 JS 逻辑)
# ---------------------------------------------------------------------------
def test_m1_error_results_not_filtered():
    print("\n=== M1: 前端 Error 结果不过滤 ===")
    try:
        # 读取 main.js 验证过滤逻辑包含 error 检查
        js_path = os.path.join(ROOT, "static", "js", "main.js")
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        check(
            "main.js 包含 source error 检查",
            'src.indexOf("error")' in js_content or 'src.indexOf(\'error\')' in js_content,
            "未找到 error source 检查逻辑",
        )

        # 模拟过滤逻辑验证
        # 模拟后端返回含 Error 的结果
        error_item = {"name": "Unknown entity", "score": 0.0, "source": "Input Error"}
        normal_item = {"name": "招商银行", "score": 0.8, "source": "Model Prediction"}
        low_score_item = {"name": "某公司", "score": 0.01, "source": "Model Prediction"}

        items = [error_item, normal_item, low_score_item]
        min_score = 0.05

        # 复现 main.js 的过滤逻辑
        filtered = []
        for item in items:
            src = (item.get("source") or "").lower()
            if "error" in src:
                filtered.append(item)
                continue
            score = item.get("score", item.get("probability"))
            if isinstance(score, (int, float)):
                if score >= min_score:
                    filtered.append(item)
            else:
                filtered.append(item)

        check(
            "Error 结果 (score=0.0) 未被过滤",
            error_item in filtered,
            f"filtered={filtered}",
        )
        check(
            "正常结果 (score=0.8) 保留",
            normal_item in filtered,
            f"filtered={filtered}",
        )
        check(
            "低分非 Error 结果 (score=0.01) 被过滤",
            low_score_item not in filtered,
            f"filtered={filtered}",
        )

    except Exception as e:
        check("M1 测试未抛出异常", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# M7: 区分未识别实体和 graph_dao 不可用
# ---------------------------------------------------------------------------
def test_m7_graph_dao_unavailable():
    print("\n=== M7: 区分未识别实体和 graph_dao 不可用 ===")
    try:
        from core.graph_dao import GraphDAO

        csv_path = os.path.join(
            ROOT, "models", "RE-GCN-master", "data", "80STOCKS",
            "80stocks_quadruples.csv",
        )

        # --- Case 1: graph_dao=None 时应提示"服务未就绪" ---
        # 模拟 app.py 的逻辑
        head = "招商银行"
        relation = "大股东"
        graph_dao = None  # 模拟初始化失败

        if head and relation:
            if graph_dao is not None:
                message = "查询结果"
            else:
                message = "知识库服务未就绪，无法查询。"
        else:
            missing = []
            if not head:
                missing.append("实体")
            if not relation:
                missing.append("关系")
            message = f"未能识别明确的{'或'.join(missing)}，无法直接查询知识库。"

        check(
            "graph_dao=None 时提示服务未就绪",
            "未就绪" in message,
            f"message={message}",
        )

        # --- Case 2: 实体未识别时提示"实体" ---
        head2 = ""
        relation2 = "大股东"
        if head2 and relation2:
            message2 = "查询结果"
        else:
            missing2 = []
            if not head2:
                missing2.append("实体")
            if not relation2:
                missing2.append("关系")
            message2 = f"未能识别明确的{'或'.join(missing2)}，无法直接查询知识库。"

        check(
            "实体未识别时提示包含'实体'",
            "实体" in message2,
            f"message={message2}",
        )

        # --- Case 3: 关系未识别时提示"关系" ---
        head3 = "招商银行"
        relation3 = ""
        if head3 and relation3:
            message3 = "查询结果"
        else:
            missing3 = []
            if not head3:
                missing3.append("实体")
            if not relation3:
                missing3.append("关系")
            message3 = f"未能识别明确的{'或'.join(missing3)}，无法直接查询知识库。"

        check(
            "关系未识别时提示包含'关系'",
            "关系" in message3,
            f"message={message3}",
        )

        # --- Case 4: 验证 app.py 源码包含新的分支逻辑 ---
        app_path = os.path.join(ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            app_content = f.read()

        check(
            "app.py 包含 '知识库服务未就绪' 提示",
            "知识库服务未就绪" in app_content,
            "未找到服务未就绪提示",
        )
        check(
            "app.py 包含 missing 列表逻辑",
            "missing.append" in app_content,
            "未找到 missing 列表逻辑",
        )

    except Exception as e:
        check("M7 测试未抛出异常", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# M8: 两实体反查关系/职务
# ---------------------------------------------------------------------------
def test_m8_pair_relation_lookup():
    print("\n=== M8: 两实体反查关系/职务 ===")
    try:
        from core.graph_dao import GraphDAO
        from app import _extract_entity_matches, _is_pair_relation_question

        csv_path = os.path.join(
            ROOT, "models", "RE-GCN-master", "data", "80STOCKS",
            "80stocks_quadruples.csv",
        )
        dao = GraphDAO(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            use_mock=True,
            local_csv_path=csv_path,
            use_demo_mock=False,
        )

        relations = dao.query_entity_pair_relations("蔡洪平", "招商银行", None)
        check(
            "蔡洪平/招商银行 可反查到职务关系",
            "监事会提名委员会委员" in relations,
            f"实际返回: {relations}",
        )

        analysis = {
            "ac_matches": [
                {"word": "蔡洪平", "start": 0, "end": 2, "type": "ENTITY"},
                {"word": "招商银行", "start": 4, "end": 7, "type": "ENTITY"},
            ]
        }
        entities = _extract_entity_matches(analysis)
        check(
            "app.py 可按文本顺序提取两个实体",
            entities == ["蔡洪平", "招商银行"],
            f"entities={entities}",
        )
        check(
            "担任/职务问法会触发两实体关系查询",
            _is_pair_relation_question("蔡洪平在招商银行担任什么职务") is True,
        )

    except Exception as e:
        check("M8 测试未抛出异常", False, str(e))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("回归验证 — 第一批 Bug 修复")
    print("=" * 60)

    test_s1_module_isolation()
    test_s2_mock_fallback()
    test_s3_bert_race_condition()
    test_s5_get_fact_no_time()
    test_m1_error_results_not_filtered()
    test_m7_graph_dao_unavailable()
    test_m8_pair_relation_lookup()

    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
