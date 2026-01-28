import sys
import os
import threading
from flask import Flask, render_template, request, jsonify
from config import Config
from core.preprocessing import NLPProcessor
from core.graph_dao import GraphDAO
from core.reasoning import ReasoningEngine
import logging

# Initialize Flask app
if getattr(sys, 'frozen', False):
    # PyInstaller mode
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    # Normal development mode
    app = Flask(__name__)

app.config.from_object(Config)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global State for Async Loading
class GlobalState:
    def __init__(self):
        self.loading = True
        self.message = "系统启动中..."
        self.nlp_processor = None
        self.graph_dao = None
        self.reasoning_engine = None

state = GlobalState()

def init_system_background():
    """Background thread to initialize heavy models."""
    global state
    with app.app_context():
        try:
            logger.info("Starting background initialization...")
            state.message = "正在加载 NLP 处理器与 BERT 模型..."
            state.nlp_processor = NLPProcessor(app.config)
            
            state.message = "正在连接图数据库..."
            state.graph_dao = GraphDAO(
                app.config['NEO4J_URI'],
                app.config['NEO4J_USER'],
                app.config['NEO4J_PASSWORD'],
                use_mock=app.config['USE_MOCK_GRAPH'],
                local_csv_path=app.config.get('GRAPH_LOCAL_CSV_PATH')
            )
            
            state.message = "正在加载知识图谱推理引擎 (REGCN)..."
            state.reasoning_engine = ReasoningEngine(app.config)
            
            state.message = "系统准备就绪"
            state.loading = False
            logger.info("Background initialization complete.")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            state.message = f"系统初始化失败: {str(e)}"
            # Don't set loading=False to keep error visible or handle differently

# Start initialization thread
init_thread = threading.Thread(target=init_system_background)
init_thread.daemon = True
init_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        "loading": state.loading,
        "message": state.message
    })

@app.route('/api/query', methods=['POST'])
def query():
    """
    Main QA endpoint.
    Orchestrates NLP -> Graph Query -> Reasoning
    """
    if state.loading:
        return jsonify({"error": "System is still initializing", "message": state.message}), 503
        
    data = request.json
    question = data.get('question')
    mode = data.get('mode', 'internal') # internal or external
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    logger.info(f"Received query: {question} [Mode: {mode}]")
    
    # 1. NLP Analysis
    analysis = state.nlp_processor.analyze(question)
    structured = analysis['structured_query']
    
    h = structured.get('h')
    r = structured.get('r')
    t = structured.get('t') # usually None for questions
    time = structured.get('time')
    
    response_data = {
        "analysis": analysis,
        "graph_result": [],
        "graph_message": "",
        "reasoning_result": []
    }
    
    # 2. Graph Lookup (if entity and relation found)
    if h and r:
        graph_result = state.graph_dao.query_entity_relation(h, r, time)
        response_data['graph_result'] = graph_result
        if not graph_result:
            response_data['graph_message'] = "暂无数据"
    else:
        response_data['graph_message'] = "未能识别明确的实体或关系，无法直接查询知识库。"

    # 3. Reasoning
    if mode == 'internal':
        # Internal Reasoning (Link Prediction)
        # Predict tail if missing
        if h and r and not t:
            exact_facts = []
            if not response_data.get('graph_result'):
                exact_facts = state.reasoning_engine.get_fact(h, r, time)
            if exact_facts:
                response_data['graph_result'] = exact_facts
                response_data['graph_message'] = ""
            
            # 2. Always run Model Prediction for Reasoning Result
            predictions = state.reasoning_engine.internal_reasoning(h, r, time)
            response_data['reasoning_result'] = predictions
            
    elif mode == 'external':
        # External Reasoning (Future/Event)
        if h:
            predictions = state.reasoning_engine.external_reasoning(h, time)
            response_data['reasoning_result'] = [
                {"name": p[0], "score": p[1]} for p in predictions
            ]
            
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
