import logging
import random
from core.regcn_wrapper import REGCNWrapper

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Handles Internal Reasoning (Link Prediction) and External Reasoning (Future Event Prediction).
    """
    def __init__(self, config):
        self.config = config
        self.use_mock = config.get('USE_MOCK_MODELS', True)
        
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
                ("陆金海", 0.98),
                ("姜国华", 0.85),
                ("郭田勇", 0.76),
                ("张三", 0.45),
                ("李四", 0.30)
            ]
            # Filter or adjust based on input to make it look dynamic
            if "贵州茅台" in str(head):
                return candidates
            else:
                return [("未知实体", 0.5)]
        
        # Real model inference
        try:
            predictions = self.regcn_model.predict(head, relation, time)
            # Adapt format to list of tuples
            # predictions is list of dicts: {"name": ..., "score": ..., "source": ...}
            # We should return list of dicts to keep 'source'
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
            # Return future event predictions
            events = [
                ("股价上涨概率", "85%"),
                ("政策风险等级", "低"),
                ("市场关注度", "高"),
                ("行业排名预测", "Top 1"),
                ("分红预期", "增加")
            ]
            return events
            
        return []
