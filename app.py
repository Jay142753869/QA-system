import sys
import os
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

# Core Modules
nlp_processor = None
graph_dao = None
reasoning_engine = None

def init_system():
    global nlp_processor, graph_dao, reasoning_engine
    if nlp_processor is None:
        logger.info("Initializing System Modules...")
        nlp_processor = NLPProcessor(app.config)
        graph_dao = GraphDAO(
            app.config['NEO4J_URI'],
            app.config['NEO4J_USER'],
            app.config['NEO4J_PASSWORD'],
            use_mock=app.config['USE_MOCK_MODELS']
        )
        reasoning_engine = ReasoningEngine(app.config)
        logger.info("System Modules Initialized.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """
    Main QA endpoint.
    Orchestrates NLP -> Graph Query -> Reasoning
    """
    if not nlp_processor:
        init_system()
        
    data = request.json
    question = data.get('question')
    mode = data.get('mode', 'internal') # internal or external
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    logger.info(f"Received query: {question} [Mode: {mode}]")
    
    # 1. NLP Analysis
    analysis = nlp_processor.analyze(question)
    structured = analysis['structured_query']
    
    h = structured.get('h')
    r = structured.get('r')
    t = structured.get('t') # usually None for questions
    time = structured.get('time')
    
    response_data = {
        "analysis": analysis,
        "graph_result": [],
        "reasoning_result": []
    }
    
    # 2. Graph Lookup (if entity and relation found)
    if h and r:
        graph_result = graph_dao.query_entity_relation(h, r, time)
        response_data['graph_result'] = graph_result
    else:
        response_data['graph_result'] = ["未能识别明确的实体或关系，无法直接查询知识库。"]

    # 3. Reasoning
    if mode == 'internal':
        # Internal Reasoning (Link Prediction)
        # Predict tail if missing
        if h and r and not t:
            predictions = reasoning_engine.internal_reasoning(h, r, time)
            response_data['reasoning_result'] = [
                {"name": p[0], "score": f"{p[1]:.2f}"} for p in predictions
            ]
    elif mode == 'external':
        # External Reasoning (Future/Event)
        if h:
            predictions = reasoning_engine.external_reasoning(h, time)
            response_data['reasoning_result'] = [
                {"name": p[0], "score": p[1]} for p in predictions
            ]
            
    return jsonify(response_data)

if __name__ == '__main__':
    with app.app_context():
        init_system()
    app.run(debug=True, port=5000)
