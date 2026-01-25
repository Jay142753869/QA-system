import ahocorasick
import jieba
import jieba.posseg as pseg
import logging
import numpy as np
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for transformers
try:
    from transformers import BertTokenizer, BertModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not found. BERT model will run in mock mode.")

class ACMatcher:
    """
    Aho-Corasick Automaton for fast multi-pattern matching.
    Used for recognizing known entities and relations in the query.
    """
    def __init__(self):
        self.automaton = ahocorasick.Automaton()
        self.built = False
        self.keywords = set()
    
    def add_keywords(self, keywords):
        """Add a list of keywords to the automaton."""
        for key in keywords:
            if key not in self.keywords:
                self.automaton.add_word(key, key)
                self.keywords.add(key)
    
    def build(self):
        """Finalize the automaton construction."""
        self.automaton.make_automaton()
        self.built = True
        logger.info(f"AC Automaton built with {len(self.keywords)} keywords.")
        
    def search(self, text):
        """
        Search for keywords in text.
        Returns list of (keyword, start_index, end_index)
        """
        if not self.built:
            logger.warning("AC Automaton not built yet.")
            return []
        
        results = []
        # iter returns (end_index, value)
        for end_index, value in self.automaton.iter(text):
            start_index = end_index - len(value) + 1
            results.append({
                "word": value,
                "start": start_index,
                "end": end_index,
                "type": "AC_MATCH" # Placeholder type
            })
        return results

class SimilarityModel:
    """
    Handles text embedding using BERT or a Mock fallback.
    """
    def __init__(self, model_name='bert-base-chinese', use_mock=True):
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None
        
        if not use_mock and TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading BERT model: {model_name}...")
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.model = BertModel.from_pretrained(model_name)
                logger.info("BERT model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load BERT model '{model_name}': {e}. Falling back to mock mode.")
                self.use_mock = True
        else:
            self.use_mock = True
            
    def encode(self, text):
        """Returns a vector representation of the text."""
        if self.use_mock:
            # Generate a deterministic pseudo-random vector based on text hash
            # This ensures same text gets same vector in mock mode
            seed = hash(text) % (2**32)
            rng = np.random.default_rng(seed)
            return rng.random(768) # Standard BERT size
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use CLS token or mean pooling
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            return np.zeros(768)

class NLPProcessor:
    """
    Main Preprocessing Module Coordinator.
    Integrates AC Automaton, Segmentation, and Embedding.
    """
    def __init__(self, config):
        self.config = config
        self.ac_matcher = ACMatcher()
        self.sim_model = SimilarityModel(use_mock=config.get('USE_MOCK_MODELS', True))
        
        # Load initial knowledge base for AC Automaton
        # In a real scenario, this would load from Neo4j or a file
        self._load_initial_data()
        
    def _load_initial_data(self):
        # Sample entities from the domain
        entities = [
            "贵州茅台", "五粮液", "招商银行", "平安银行", "浦发银行", # Companies
            "陆金海", "姜国华", "郭田勇", # People
            "大股东", "独立董事", "董事长", "净利润", "营业收入", # Relations/Attributes
            # "2024年", "2023年", "2022年", "5月", "12月" # Time - Now handled by dynamic regex
        ]
        self.ac_matcher.add_keywords(entities)
        self.ac_matcher.build()
        
    def analyze(self, text):
        """
        Full analysis pipeline:
        1. AC Match (Known Entities/Relations)
        2. Dynamic Regex Match (Time/Dates)
        3. Segmentation & POS Tagging (Jieba - Fallback for Unknown Entities)
        4. Entity Linking & Intent Classification (Simulated)
        """
        logger.info(f"Analyzing query: {text}")
        
        # Initialize Quadruple
        quadruple = {"h": None, "r": None, "t": None, "time": None}
        enriched_matches = []
        
        # 1. AC Automaton Match (Priority 1: Known Knowledge)
        ac_matches = self.ac_matcher.search(text)
        
        for match in ac_matches:
            word = match['word']
            match_type = "UNKNOWN"
            
            # Heuristic mapping for AC matches
            if word in ["大股东", "独立董事", "董事长", "净利润", "营业收入"]:
                quadruple['r'] = word
                match_type = "RELATION"
            else:
                quadruple['h'] = word # Assume entity
                match_type = "ENTITY"
            
            enriched_matches.append({
                "word": word,
                "start": match['start'],
                "end": match['end'],
                "type": match_type
            })

        # 2. Dynamic Regex Match (Priority 2: Time)
        # Match patterns like: 2018年, 2024年, 5月, 2024-05
        time_patterns = [
            r"(\d{4}年)",       # 2018年
            r"(\d{1,2}月)",     # 5月
            r"(\d{4}-\d{1,2})", # 2024-05
            r"(\d{4})"          # 2018 (Context dependent, usually year)
        ]
        
        for pattern in time_patterns:
            for m in re.finditer(pattern, text):
                word = m.group()
                # If we haven't found a time yet, or this is more specific, take it
                if not quadruple['time']:
                    quadruple['time'] = word
                elif word not in quadruple['time']: 
                    # Append if different (e.g. 2018年 5月)
                    quadruple['time'] += word
                
                # Check if this overlaps with existing matches to avoid duplication
                is_new = True
                for existing in enriched_matches:
                    if existing['start'] == m.start() + 1: # AC index is 1-based in my wrapper? Wait, let's check wrapper
                        # AC wrapper: start_index = end_index - len(value) + 1. So it is 0-based index if end_index is 0-based?
                        # ahocorasick .iter() returns (end_index, value). end_index is 0-based index of the last character.
                        # So if "abc" (012), end_index=2. start = 2 - 3 + 1 = 0. Correct.
                        pass
                
                enriched_matches.append({
                    "word": word,
                    "start": m.start(), # 0-based
                    "end": m.end() - 1, # 0-based inclusive
                    "type": "TIME"
                })

        # 3. Jieba Segmentation (Priority 3: Fallback for Unknown Entities)
        words = pseg.cut(text)
        seg_results = []
        for w, f in words:
            seg_results.append({"word": w, "flag": f})
            
            # Fallback Logic: If no Entity found yet, look for Nouns (nr, ns, nt, nz)
            if not quadruple['h']:
                if f in ['nr', 'ns', 'nt', 'nz']: # Person, Location, Organization, Other Proper Noun
                    # Double check it's not a Relation or Time we already found
                    is_known = False
                    for m in enriched_matches:
                        if m['word'] == w:
                            is_known = True
                            break
                    
                    if not is_known:
                        quadruple['h'] = w
                        enriched_matches.append({
                            "word": w,
                            "start": -1, # Unknown position from Jieba stream
                            "end": -1,
                            "type": "ENTITY (Inferred)"
                        })

        # 4. Get Embedding
        embedding = self.sim_model.encode(text)
        
        return {
            "original_text": text,
            "ac_matches": enriched_matches,
            "segmentation": seg_results,
            "structured_query": quadruple,
            "embedding_sample": embedding[:5].tolist() # Just show first 5 dims
        }
