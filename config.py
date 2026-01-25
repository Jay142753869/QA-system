import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    
    # Neo4j Configuration (Placeholder)
    NEO4J_URI = os.environ.get('NEO4J_URI') or "bolt://localhost:7687"
    NEO4J_USER = os.environ.get('NEO4J_USER') or "neo4j"
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD') or "password"
    
    # Model Configuration
    # Set to True to use Mock models (since real models are not trained yet)
    USE_MOCK_MODELS = True
    
    # Path to vocabulary or entities file (if needed)
    DATA_DIR = os.path.join(os.getcwd(), 'data')
