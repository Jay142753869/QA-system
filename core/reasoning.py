import logging
import random
from core.regcn_wrapper import REGCNWrapper
from core.tirgn_wrapper import TiRGNWrapper

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Handles Internal Reasoning (Link Prediction) and External Reasoning (Future Event Prediction).
    """
    def __init__(self, config):
        self.config = config
        self.use_mock = config.get('USE_MOCK_MODELS', True)
        self.tirgn_model = None
        
        if not self.use_mock:
            logger.info("Initializing REGCN Model for Internal Reasoning...")
            try:
                self.regcn_model = REGCNWrapper(config)
                logger.info("REGCN Model Initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize REGCN Model: {e}. Falling back to Mock.")
                self.use_mock = True
        
    def internal_reasoning(self, head, relation, time=None):
        """
        Predict missing tails for (h, r, ?, t).
        Returns list of (candidate, probability).
        """
        logger.info(f"Internal Reasoning for: {head}, {relation}, {time}")
        
        if self.use_mock:
            # Return plausible candidates with scores
            candidates = [
                {"name": "陆金海", "score": 0.98, "source": "Mock Prediction"},
                {"name": "姜国华", "score": 0.85, "source": "Mock Prediction"},
                {"name": "郭田勇", "score": 0.76, "source": "Mock Prediction"},
                {"name": "张三", "score": 0.45, "source": "Mock Prediction"},
                {"name": "李四", "score": 0.30, "source": "Mock Prediction"},
            ]
            # Filter or adjust based on input to make it look dynamic
            if "贵州茅台" in str(head):
                return candidates
            else:
                return [{"name": "未知实体", "score": 0.5, "source": "Mock Prediction"}]
        
        # Real model inference
        try:
            predictions = self.regcn_model.predict(head, relation, time)
            return predictions
        except Exception as e:
            logger.error(f"REGCN Prediction failed: {e}")
            return [{"name": "Prediction Error", "score": 0.0, "source": "Error"}]

    def get_fact(self, head, relation, time=None):
        if not self.use_mock and self.regcn_model:
            return self.regcn_model.get_fact(head, relation, time)
        return []

    def external_reasoning(self, entity, time=None):
        """
        Predict future events or external impacts.
        Returns list of (event_description, probability).
        """
        logger.info(f"External Reasoning for: {entity}, {time}")
        
        if self.use_mock:
            return [
                {"name": "股价上涨概率", "score": "85%", "source": "Mock External"},
                {"name": "政策风险等级", "score": "低", "source": "Mock External"},
                {"name": "市场关注度", "score": "高", "source": "Mock External"},
                {"name": "行业排名预测", "score": "Top 1", "source": "Mock External"},
                {"name": "分红预期", "score": "增加", "source": "Mock External"},
            ]

        if self.tirgn_model is None:
            try:
                self.tirgn_model = TiRGNWrapper(self.config)
            except Exception as e:
                logger.error(f"Failed to initialize TiRGN: {e}")
                return [{"name": "外推模型初始化失败", "score": 0.0, "source": "TiRGN Error"}]

        return [{"name": "请输入实体与关系以进行外推", "score": 0.0, "source": "TiRGN"}]

    def external_reasoning_tirgn(self, head, relation, time=None):
        logger.info(f"External Reasoning (TiRGN) for: {head}, {relation}, {time}")
        if self.use_mock:
            return self.external_reasoning(head, time)
        if self.tirgn_model is None:
            self.tirgn_model = TiRGNWrapper(self.config)
        return self.tirgn_model.predict_tail(head, relation, time, top_k=5)
