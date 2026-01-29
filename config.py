import os
import sys

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    
    # Neo4j Configuration (Placeholder)
    NEO4J_URI = os.environ.get('NEO4J_URI') or "bolt://localhost:7687"
    NEO4J_USER = os.environ.get('NEO4J_USER') or "neo4j"
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD') or "password"
    
    # Model Configuration
    # Set to False to use Real REGCN model
    USE_MOCK_MODELS = False

    # Graph Configuration
    USE_MOCK_GRAPH = True
    
    BASE_PATH = get_base_path()
    
    # Path to vocabulary or entities file (if needed)
    DATA_DIR = os.path.join(BASE_PATH, 'data')

    # REGCN Configuration
    REGCN_BASE_DIR = os.path.join(BASE_PATH, 'models', 'RE-GCN-master')
    REGCN_DATASET = '80STOCKS'
    REGCN_MODEL_FILE = '80STOCKS-uvrgcn-convtranse-ly2-dilate1-his3-weight_0.5-discount_1.0-angle_10-dp0.2_0.2_0.2_0.2-gpu0'
    REGCN_MODEL_PATH = os.path.join(REGCN_BASE_DIR, 'models', REGCN_MODEL_FILE)
    REGCN_DATA_DIR = os.path.join(REGCN_BASE_DIR, 'data', REGCN_DATASET)

    GRAPH_LOCAL_CSV_PATH = os.path.join(REGCN_DATA_DIR, '80stocks_quadruples.csv')
    
    # REGCN Hyperparameters (extracted from filename)
    REGCN_PARAMS = {
        'encoder': 'uvrgcn',
        'decoder': 'convtranse',
        'n_layers': 2,
        'train_history_len': 3,
        'dilate_len': 1,
        'weight': 0.5,
        'discount': 1.0,
        'angle': 10,
        'dropout': 0.2,
        'input_dropout': 0.2,
        'hidden_dropout': 0.2,
        'feat_dropout': 0.2,
        'gpu': 0, # Will check availability dynamically
        'n_hidden': 200, # Default from main.py
        'n_bases': 100,  # Default from main.py
        'n_basis': 100,  # Default from main.py
        'opn': 'sub',    # Default from main.py
        'self_loop': True,
        'skip_connect': False,
        'layer_norm': False,
        'aggregation': 'none',
        'entity_prediction': True,
        'relation_prediction': False,
        'add_static_graph': False,
        'run_analysis': False
    }

    # TiRGN Configuration (External Reasoning)
    TIRGN_BASE_DIR = os.path.join(BASE_PATH, 'models', 'TiRGN-main')
    TIRGN_DATASET = '80STOCKS'
    TIRGN_MODEL_FILE = 'gl_rate_0.3-80STOCKS-convgcn-timeconvtranse-ly2-dilate1-his9-weight_0.5-discount_1.0-angle_14-dp0.2_0.2_0.2_0.2-gpu0-checkpoint'
    TIRGN_MODEL_PATH = os.path.join(TIRGN_BASE_DIR, 'models', TIRGN_MODEL_FILE)
    TIRGN_DATA_DIR = os.path.join(TIRGN_BASE_DIR, 'data', TIRGN_DATASET)
    TIRGN_HISTORY_DIR = os.path.join(TIRGN_DATA_DIR, 'history')

    TIRGN_PARAMS = {
        'encoder': 'convgcn',
        'decoder': 'timeconvtranse',
        'n_layers': 2,
        'train_history_len': 9,
        'dilate_len': 1,
        'history_rate': 0.3,
        'weight': 0.5,
        'discount': 1.0,
        'angle': 14,
        'dropout': 0.2,
        'input_dropout': 0.2,
        'hidden_dropout': 0.2,
        'feat_dropout': 0.2,
        'gpu': 0,
        'n_hidden': 200,
        'n_bases': 100,
        'n_basis': 100,
        'opn': 'sub',
        'self_loop': True,
        'skip_connect': False,
        'aggregation': 'none',
        'num_times': 380,
        'time_interval': 1,
        'use_static': True,
        'run_analysis': False,
    }
