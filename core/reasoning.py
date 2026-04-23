import logging
import random
import threading

from core.regcn_wrapper import REGCNWrapper
from core.tirgn_wrapper import TiRGNWrapper

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Handles Internal Reasoning (Link Prediction) and External Reasoning (Future Event Prediction).
    """

    def __init__(self, config):
        self.config = config
        self.regcn_model = None
        self.tirgn_model = None
        self._regcn_lock = threading.Lock()
        self._tirgn_lock = threading.Lock()

    def _ensure_regcn_model(self):
        """Lazily initialize the REGCN model in a thread-safe way."""
        if self.regcn_model is not None:
            return self.regcn_model

        with self._regcn_lock:
            if self.regcn_model is None:
                logger.info("Initializing REGCN Model for Internal Reasoning...")
                self.regcn_model = REGCNWrapper(self.config)
                logger.info("REGCN Model Initialized.")
        return self.regcn_model

    def _ensure_tirgn_model(self):
        """Lazily initialize the TiRGN model in a thread-safe way."""
        if self.tirgn_model is not None:
            return self.tirgn_model

        with self._tirgn_lock:
            if self.tirgn_model is None:
                logger.info("Initializing TiRGN Model for External Reasoning...")
                self.tirgn_model = TiRGNWrapper(self.config)
                logger.info("TiRGN Model Initialized.")
        return self.tirgn_model

    def internal_reasoning(self, head, relation, time=None, top_k=5):
        """
        Predict missing tails for (h, r, ?, t).
        Returns list of (candidate, probability).
        """
        logger.info(f"Internal Reasoning for: {head}, {relation}, {time}")

        try:
            model = self._ensure_regcn_model()
        except Exception as e:
            logger.error(f"Failed to initialize REGCN Model: {e}")
            return [{"name": f"REGCN initialization failed: {e}", "score": 0.0, "source": "REGCN Error"}]

        try:
            return model.predict(head, relation, time, top_k=top_k)
        except Exception as e:
            logger.error(f"REGCN Prediction failed: {e}")
            return [{"name": "Prediction Error", "score": 0.0, "source": "Error"}]

    def get_fact(self, head, relation, time=None):
        try:
            model = self._ensure_regcn_model()
        except Exception as e:
            logger.error(f"Failed to initialize REGCN Model while retrieving fact: {e}")
            return []
        return model.get_fact(head, relation, time)

    def external_reasoning(self, entity, time=None):
        """
        Predict future events or external impacts.
        Returns list of (event_description, probability).
        """
        logger.info(f"External Reasoning for: {entity}, {time}")

        try:
            self._ensure_tirgn_model()
        except Exception as e:
            logger.error(f"Failed to initialize TiRGN: {e}")
            return [{"name": f"TiRGN initialization failed: {e}", "score": 0.0, "source": "TiRGN Error"}]

        return [{"name": "Please provide both entity and relation for external reasoning.", "score": 0.0, "source": "TiRGN"}]

    def external_reasoning_tirgn(self, head, relation, time=None, top_k=5):
        logger.info(f"External Reasoning (TiRGN) for: {head}, {relation}, {time}")
        try:
            model = self._ensure_tirgn_model()
        except Exception as e:
            logger.error(f"Failed to initialize TiRGN Model: {e}")
            return [{"name": f"TiRGN initialization failed: {e}", "score": 0.0, "source": "TiRGN Error"}]
        try:
            return model.predict_tail(head, relation, time, top_k=top_k)
        except Exception as e:
            logger.error(f"TiRGN Prediction failed: {e}")
            return [{"name": "TiRGN prediction failed", "score": 0.0, "source": "TiRGN Error"}]
