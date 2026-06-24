# Agent Coding Guidelines

> For AI agents working in this Financial Event Question Answering System codebase.

## Build/Lint/Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web service
python app.py

# Run desktop GUI
python gui_launcher.py

# Run model training
python train_model.py

# Run integration tests (standalone scripts)
python test_regcn_integration.py
python test_tirgn_integration.py
```

**Note:** No pytest/tox/unittest configured. Tests are standalone scripts.

## Code Style Guidelines

### Import Ordering
1. Standard library imports first
2. Third-party imports (Flask, torch, numpy, etc.)
3. Local application imports (from core import ...)

Example:
```python
import os
import sys
import json
import logging

from flask import Flask, request, jsonify
import torch
import numpy as np

from core import graph_dao, preprocessing
from config import Config
```

### Naming Conventions
- **Classes**: PascalCase (`RegcnWrapper`, `GraphDAO`)
- **Functions/Methods**: snake_case (`get_entity_relations`, `load_model`)
- **Variables**: snake_case (`entity_list`, `model_path`)
- **Constants**: UPPER_SNAKE_CASE (in config.py: `NEO4J_URI`, `USE_MOCK_GRAPH`)
- **Private methods**: leading underscore (`_load_config`, `_preprocess`)

### Type Hints
- Use type hints for function parameters and return values where clarity helps
- Common patterns observed:
```python
def predict(self, entity: str, relation: str, top_k: int = 5) -> list:
    ...

def load_model(self, model_path: str) -> torch.nn.Module:
    ...
```

### Docstrings
- Use triple-quoted docstrings for modules, classes, and functions
- First line is a brief summary
- Chinese comments/docstrings are acceptable for domain-specific financial terms

Example:
```python
def recognize_time(self, text: str) -> dict:
    """
    识别文本中的时间信息

    Args:
        text: 用户输入的查询文本

    Returns:
        dict: 包含year, month, day的字典，未识别则为None
    """
```

### Error Handling
- Use try-except blocks with specific exceptions
- Log errors using logging module, not print statements
- Provide fallback behavior when possible

Pattern:
```python
import logging
logger = logging.getLogger(__name__)

try:
    result = self.model.predict(input_data)
except Exception as e:
    logger.error(f"Model prediction failed: {e}")
    result = self._fallback_prediction(input_data)
```

### Class Patterns
- Use `__init__` for initialization with dependency injection when appropriate
- Use `@staticmethod` or `@classmethod` for utility methods that don't need self
- Private methods prefixed with underscore

Example:
```python
class RegcnWrapper:
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path or Config.REGCN_MODEL_PATH
        self._load_model()

    def _load_model(self):
        """Private method to load model weights"""
        ...
```

### Configuration
- All configuration lives in `config.py`
- Use environment variables for sensitive data (passwords, URIs)
- Provide sensible defaults for local development
- Use boolean flags like `USE_MOCK_GRAPH` and `USE_MOCK_MODELS` to toggle between real and mock implementations

### Testing
- Tests are standalone Python scripts, not using pytest
- Integration tests test real model loading and prediction
- Use descriptive test function names
- Print or log results for manual verification

### Code Organization
- `core/`: Business logic (preprocessing, graph access, reasoning)
- `models/`: Third-party model implementations (RE-GCN, TiRGN)
- `static/` + `templates/`: Flask web UI
- Root level: Entry points (app.py, gui_launcher.py, train_model.py)

### Bilingual Considerations
- Financial domain terms are in Chinese (entities like "招商银行", relations like "监事会提名委员会委员")
- Code, comments, and docstrings can be in Chinese for domain clarity
- User-facing strings should match the target audience language

## Key Libraries & Frameworks
- **Web**: Flask (REST API + server-side rendering)
- **ML/DL**: PyTorch, transformers (BERT), numpy
- **NLP**: jieba (Chinese segmentation), ahocorasick (AC automaton)
- **Graph**: neo4j (graph database)
- **GUI**: pywebview (desktop wrapper)
- **Data**: pandas (CSV processing)
- **Utilities**: python-dotenv (env vars), tqdm (progress bars)
