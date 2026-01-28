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

    def encode_list(self, text_list):
        """Batch encode a list of texts."""
        if self.use_mock:
            return [self.encode(t) for t in text_list]
        
        embeddings = []
        # Process one by one for simplicity (or batch if needed)
        # For small relation list (few hundreds), one by one is fine for initialization
        for text in text_list:
            embeddings.append(self.encode(text))
        return embeddings

    def compute_similarity(self, query_emb, candidate_embs):
        """
        Compute cosine similarity between query embedding and a list of candidate embeddings.
        Returns indices sorted by similarity (descending).
        """
        # query_emb: (D,)
        # candidate_embs: (N, D)
        
        # Normalize
        norm_q = np.linalg.norm(query_emb)
        if norm_q == 0: return []
        query_emb = query_emb / norm_q
        
        candidate_matrix = np.array(candidate_embs)
        norm_c = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        norm_c[norm_c == 0] = 1
        candidate_matrix = candidate_matrix / norm_c
        
        # Dot product
        scores = np.dot(candidate_matrix, query_emb)
        
        # Sort indices
        sorted_indices = np.argsort(scores)[::-1]
        return sorted_indices, scores[sorted_indices]

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
        
        # BERT Relation Extraction Setup
        self.relation_list = []
        self.relation_embs = []
        self._load_relations()

    def _load_relations(self):
        """Load relations from relation2id.txt and compute their embeddings."""
        rel_file = os.path.join(self.config.get('REGCN_DATA_DIR', ''), 'relation2id.txt')
        if not os.path.exists(rel_file):
            # Fallback path logic or hardcoded typical path
            rel_file = os.path.join(os.getcwd(), 'models', 'RE-GCN-master', 'data', '80STOCKS', 'relation2id.txt')
        
        if os.path.exists(rel_file):
            logger.info(f"Loading relations from {rel_file}...")
            with open(rel_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        self.relation_list.append(parts[0])
            
            # Pre-compute embeddings
            logger.info(f"Computing BERT embeddings for {len(self.relation_list)} relations...")
            self.relation_embs = self.sim_model.encode_list(self.relation_list)
            logger.info("Relation embeddings computed.")
        else:
            logger.warning("relation2id.txt not found. BERT Relation Extraction will be disabled.")
        
    def _load_initial_data(self):
        """Load entities from entity2id.txt and relations from relation2id.txt"""
        # 1. Load entities from entity2id.txt
        ent_file = os.path.join(self.config.get('REGCN_DATA_DIR', ''), 'entity2id.txt')
        if not os.path.exists(ent_file):
            # Fallback path
            ent_file = os.path.join(os.getcwd(), 'models', 'RE-GCN-master', 'data', '80STOCKS', 'entity2id.txt')
        
        entities_to_add = []
        if os.path.exists(ent_file):
            logger.info(f"Loading entities from {ent_file}...")
            with open(ent_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        # parts[0] is the entity name
                        entities_to_add.append(parts[0])
            logger.info(f"Loaded {len(entities_to_add)} entities.")
        else:
            logger.warning("entity2id.txt not found. Using default sample entities.")
            # Sample entities from the domain (Fallback)
            entities_to_add = [
                "贵州茅台", "五粮液", "招商银行", "平安银行", "浦发银行", "万华化学",
                "陆金海", "姜国华", "郭田勇"
            ]

        rel_file = os.path.join(self.config.get('REGCN_DATA_DIR', ''), 'relation2id.txt')
        if not os.path.exists(rel_file):
            rel_file = os.path.join(os.getcwd(), 'models', 'RE-GCN-master', 'data', '80STOCKS', 'relation2id.txt')

        relations = []
        if os.path.exists(rel_file):
            with open(rel_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1 and parts[0]:
                        relations.append(parts[0])
        relations.extend(["大股东", "独立董事"])
        
        # Add all to AC Matcher
        self.ac_matcher.add_keywords(entities_to_add)
        self.ac_matcher.add_keywords(relations)
        
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
        raw_matches = self.ac_matcher.search(text)
        
        # Filter overlapping matches (Keep Longest Match Principle)
        # Sort by length descending to prioritize longer matches (e.g. "交通银行" > "银行")
        raw_matches.sort(key=lambda x: len(x['word']), reverse=True)
        
        final_ac_matches = []
        occupied_indices = set()
        
        for match in raw_matches:
            start = match['start']
            end = match['end'] # Inclusive 0-based indices
            
            # Check for overlap with already selected (longer) matches
            is_overlap = False
            for i in range(start, end + 1):
                if i in occupied_indices:
                    is_overlap = True
                    break
            
            if not is_overlap:
                final_ac_matches.append(match)
                # Mark indices as occupied
                for i in range(start, end + 1):
                    occupied_indices.add(i)
        
        for match in final_ac_matches:
            word = match['word']
            match_type = "UNKNOWN"
            
            # Heuristic mapping for AC matches
            # TODO: Better distinction between entity and relation
            # Check if it's in relation_list
            if word in self.relation_list or word in ["大股东", "独立董事", "董事长", "净利润", "营业收入", "高管", "监事会提名委员会委员", "风险总监"]:
                # But wait, relation_list is loaded AFTER this method is called in __init__
                # Actually _load_relations is called after _load_initial_data
                # So self.relation_list is empty here? No, ac_matcher.search is called in analyze(), which is after __init__.
                # So self.relation_list should be populated.
                
                # Priority: If it's in our known relation list, it's a relation
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
            r"(\d{8})",
            r"(\d{4}年\d{1,2}月\d{1,2}日)",
            r"(\d{4}年\d{1,2}月)",
            r"(\d{4}-\d{1,2}-\d{1,2})",
            r"(\d{4}-\d{1,2})",
            r"(\d{4}年)",
            r"(\d{1,2}月)",
            r"(\d{4})"
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
        
        # 5. BERT-based Relation Extraction (Fallback if no relation found)
        if not quadruple['r'] and self.relation_list:
            # Extract "unknown" parts of the query to match against relations
            # Simple approach: remove detected entities and times
            remaining_text = text
            for m in enriched_matches:
                remaining_text = remaining_text.replace(m['word'], "")
            
            # Remove common stopwords/punctuation
            for stop in ["是", "谁", "的", "?", "？", "查询", "我想知道", "告诉我"]:
                remaining_text = remaining_text.replace(stop, "")
            
            remaining_text = remaining_text.strip()
            
            if remaining_text:
                logger.info(f"Attempting BERT matching for relation using text: '{remaining_text}'")
                query_emb = self.sim_model.encode(remaining_text)
                
                # Check cache logic could go here, but compute_similarity is fast for 237 items
                indices, scores = self.sim_model.compute_similarity(query_emb, self.relation_embs)
                
                if len(indices) > 0:
                    top_idx = indices[0]
                    top_score = scores[0]
                    top_rel = self.relation_list[top_idx]
                    
                    logger.info(f"BERT Top match: {top_rel} (score: {top_score:.4f})")
                    
                    # Threshold check (e.g., 0.6)
                    if top_score > 0.6: # Adjust this threshold as needed
                        quadruple['r'] = top_rel
                        enriched_matches.append({
                            "word": remaining_text, # Or the matched relation name? Let's show the matched part
                            "start": -1, # Hard to map back exactly without alignment
                            "end": -1,
                            "type": f"RELATION (BERT: {top_rel})"
                        })
        
        return {
            "original_text": text,
            "ac_matches": enriched_matches,
            "segmentation": seg_results,
            "structured_query": quadruple,
            "embedding_sample": embedding[:5].tolist() # Just show first 5 dims
        }
