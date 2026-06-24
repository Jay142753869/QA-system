import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, render_template, request

from config import Config
from core.graph_dao import GraphDAO
from core.preprocessing import NLPProcessor
from core.reasoning import ReasoningEngine


if getattr(sys, "frozen", False):
    template_folder = os.path.join(sys._MEIPASS, "templates")
    static_folder = os.path.join(sys._MEIPASS, "static")
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

app.config.from_object(Config)

# --- Logging setup ---
# Console handler (always on)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_console_handler.setFormatter(_console_fmt)

# File handler (writes to %LOCALAPPDATA%\FinancialQA\logs\app.log)
_log_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "FinancialQA", "logs")
try:
    os.makedirs(_log_dir, exist_ok=True)
    _file_handler = RotatingFileHandler(
        os.path.join(_log_dir, "app.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_console_fmt)
    logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler], force=True)
except Exception:
    # Fallback to console-only if log directory is not writable
    logging.basicConfig(level=logging.INFO, handlers=[_console_handler], force=True)

logger = logging.getLogger(__name__)


class GlobalState:
    """Thread-safe container for application startup state."""

    def __init__(self):
        self._lock = threading.RLock()
        self._loading = True
        self._message = "系统启动中..."
        self._nlp_processor = None
        self._graph_dao = None
        self._reasoning_engine = None

    @property
    def loading(self):
        with self._lock:
            return self._loading

    @loading.setter
    def loading(self, value):
        with self._lock:
            self._loading = value

    @property
    def message(self):
        with self._lock:
            return self._message

    @message.setter
    def message(self, value):
        with self._lock:
            self._message = value

    @property
    def nlp_processor(self):
        with self._lock:
            return self._nlp_processor

    @nlp_processor.setter
    def nlp_processor(self, value):
        with self._lock:
            self._nlp_processor = value

    @property
    def graph_dao(self):
        with self._lock:
            return self._graph_dao

    @graph_dao.setter
    def graph_dao(self, value):
        with self._lock:
            self._graph_dao = value

    @property
    def reasoning_engine(self):
        with self._lock:
            return self._reasoning_engine

    @reasoning_engine.setter
    def reasoning_engine(self, value):
        with self._lock:
            self._reasoning_engine = value


state = GlobalState()


def init_system_background():
    """Initialize heavy dependencies in the background."""
    with app.app_context():
        start_time = time.time()
        max_init_time = 120

        try:
            logger.info("Starting background initialization...")

            state.message = "正在加载 NLP 处理器与 BERT 模型..."
            try:
                state.nlp_processor = NLPProcessor(app.config)
                logger.info("NLP Processor initialized successfully.")
            except Exception as exc:
                logger.error("NLP Processor initialization failed: %s", exc)
                state.message = f"NLP 初始化失败: {exc}"
                state.loading = False
                return

            if time.time() - start_time > max_init_time:
                logger.error("Initialization timeout reached.")
                state.message = "系统初始化超时，请重启应用。"
                state.loading = False
                return

            state.message = "正在连接图数据库..."
            try:
                state.graph_dao = GraphDAO(
                    app.config["NEO4J_URI"],
                    app.config["NEO4J_USER"],
                    app.config["NEO4J_PASSWORD"],
                    use_mock=app.config["USE_MOCK_GRAPH"],
                    local_csv_path=app.config.get("GRAPH_LOCAL_CSV_PATH"),
                    use_demo_mock=app.config.get("USE_DEMO_MOCK_GRAPH", False),
                )
                logger.info("Graph DAO initialized successfully.")
            except Exception as exc:
                logger.error("Graph DAO initialization failed: %s", exc)
                state.message = f"图数据库连接失败，已切换到本地模式: {exc}"

            state.message = "正在加载知识图谱推理引擎..."
            try:
                state.reasoning_engine = ReasoningEngine(app.config)
                logger.info("Reasoning Engine initialized successfully.")
            except Exception as exc:
                logger.error("Reasoning Engine initialization failed: %s", exc)
                state.message = f"推理引擎初始化失败: {exc}"
                state.loading = False
                return

            elapsed = time.time() - start_time
            state.message = f"系统准备就绪（初始化用时 {elapsed:.1f} 秒）"
            state.loading = False
            logger.info("Background initialization complete in %.1f seconds.", elapsed)
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.exception("Initialization failed after %.1fs: %s", elapsed, exc)
            state.message = f"系统初始化失败: {exc}"
            state.loading = False


init_thread = threading.Thread(target=init_system_background, daemon=True)
init_thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    ready = (not state.loading) and state.nlp_processor is not None and state.reasoning_engine is not None
    return jsonify(
        {
            "loading": state.loading,
            "message": state.message,
            "ready": ready,
        }
    )


PAIR_RELATION_INTENT_TERMS = ("担任", "职务", "职位", "岗位", "关系", "之间", "是什么")


def _extract_entity_matches(analysis):
    """Return deduplicated entity matches from NLP analysis, in text order."""
    entities = []
    seen = set()
    matches = analysis.get("ac_matches", [])
    sorted_matches = sorted(matches, key=lambda item: item.get("start", 10**9))

    for match in sorted_matches:
        if match.get("type") != "ENTITY":
            continue
        word = match.get("word")
        if word and word not in seen:
            entities.append(word)
            seen.add(word)

    return entities


def _is_pair_relation_question(question):
    return any(term in question for term in PAIR_RELATION_INTENT_TERMS)


@app.route("/api/query", methods=["POST"])
def query():
    """Main QA endpoint: NLP -> Graph Query -> Reasoning."""
    try:
        if state.loading:
            return jsonify({"error": "System is still initializing", "message": state.message}), 503
        if state.nlp_processor is None or state.reasoning_engine is None:
            return jsonify({"error": "System not ready", "message": state.message}), 503

        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        mode = data.get("mode", "internal")
        top_k_raw = data.get("top_k", 5)

        if not question:
            return jsonify({"error": "No question provided"}), 400

        if mode not in ("internal", "external"):
            mode = "internal"

        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))

        logger.info("Received query: %s [Mode: %s]", question, mode)

        analysis = state.nlp_processor.analyze(question)
        structured = analysis["structured_query"]

        head = structured.get("h")
        relation = structured.get("r")
        tail = structured.get("t")
        query_time = structured.get("time")
        entities = _extract_entity_matches(analysis)

        response_data = {
            "analysis": analysis,
            "graph_result": [],
            "graph_message": "",
            "reasoning_result": [],
        }

        if head and relation:
            if state.graph_dao is not None:
                graph_result = state.graph_dao.query_entity_relation(head, relation, query_time)
                response_data["graph_result"] = graph_result
                if not graph_result:
                    response_data["graph_message"] = "暂无参考数据" if mode == "external" else "暂无数据"
                elif mode == "external":
                    response_data["graph_message"] = "知识库参考（用于对比）"
            else:
                response_data["graph_message"] = "知识库服务未就绪，无法查询。"
        elif not relation and len(entities) >= 2 and _is_pair_relation_question(question):
            if state.graph_dao is not None:
                pair_relations = state.graph_dao.query_entity_pair_relations(
                    entities[0],
                    entities[1],
                    query_time,
                )
                response_data["graph_result"] = pair_relations
                if pair_relations:
                    response_data["graph_message"] = ""
                else:
                    response_data["graph_message"] = "暂无数据"
            else:
                response_data["graph_message"] = "知识库服务未就绪，无法查询。"
        else:
            missing = []
            if not head:
                missing.append("实体")
            if not relation:
                missing.append("关系")
            response_data["graph_message"] = f"未能识别明确的{'或'.join(missing)}，无法直接查询知识库。"

        if mode == "internal":
            if head and relation and not tail:
                exact_facts = []
                if not response_data.get("graph_result"):
                    exact_facts = state.reasoning_engine.get_fact(head, relation, query_time)
                if exact_facts:
                    response_data["graph_result"] = exact_facts
                    response_data["graph_message"] = ""

                predictions = state.reasoning_engine.internal_reasoning(head, relation, query_time, top_k=top_k)
                response_data["reasoning_result"] = predictions
        elif mode == "external":
            if head and relation:
                predictions = state.reasoning_engine.external_reasoning_tirgn(
                    head, relation, query_time, top_k=top_k
                )
                response_data["reasoning_result"] = predictions
            else:
                response_data["reasoning_result"] = []
                if not head:
                    response_data["graph_message"] = "未能识别明确的实体，无法调用外推模型。"
                elif not relation:
                    response_data["graph_message"] = "未能识别明确的关系，外推模型需要实体与关系。"

        return jsonify(response_data)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        return jsonify({"error": "Internal server error", "message": "问答处理失败，请稍后重试"}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=debug_mode, port=5000, host="0.0.0.0")
