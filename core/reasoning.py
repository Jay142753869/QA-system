import logging
import random

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Handles Internal Reasoning (Link Prediction) and External Reasoning (Future Event Prediction).
    """
    def __init__(self, config):
        self.config = config
        self.use_mock = config.get('USE_MOCK_MODELS', True)
        # In the future, load TAMG model here
        # self.model = load_tamg_model(config['MODEL_PATH'])
        
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
        
        # Real model inference logic would go here
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
