import ahocorasick
import jieba
import jieba.posseg as pseg
import logging
import numpy as np
import os
import re
import hashlib
import threading

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


def get_cfg(config, key, default=None):
    if hasattr(config, 'get'):
        val = config.get(key)
        if val is not None:
            return val
    return getattr(config, key, default)

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
    BERT model is loaded lazily (on first use) to speed up startup.
    """
    def __init__(self, model_name='bert-base-chinese', use_mock=True):
        self.use_mock = use_mock
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._bert_loaded = False
        self._lock = threading.Lock()
        self._load_event = threading.Event()
        self._load_event.set()  # Initially "no load in progress"
        self._load_generation = 0  # Invalidates stale background loads
        
        # Note: BERT is now loaded lazily in _ensure_bert_loaded()
        if not use_mock and not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. BERT will run in mock mode.")
            self.use_mock = True
    
    def _ensure_bert_loaded(self, timeout=30):
        """
        Lazy load BERT model with timeout.
        Returns True if BERT is ready (or mock mode), False on timeout.

        Uses a Lock + Event + generation-token pattern so that:
        - Only one background load runs at a time.
        - Callers waiting on the Event are released atomically when
          the background thread commits its result.
        - A stale background thread (superseded by a newer load) cannot
          overwrite a model that is already in use.
        - ``encode()`` never sees a half-loaded tokenizer/model pair
          because both are committed together under the lock.
        """
        if self.use_mock:
            return True
        
        with self._lock:
            if self._bert_loaded:
                # Already resolved (success or fallback to mock).
                return True
            if not self._load_event.is_set():
                # A load is already in progress; just wait for it.
                pass
            else:
                # Start a new background load.
                self._load_event.clear()
                self._load_generation += 1
                my_generation = self._load_generation
                t = threading.Thread(
                    target=self._load_bert_worker,
                    args=(my_generation,),
                    daemon=True,
                )
                t.start()

        # Wait (outside the lock) for the background thread to finish.
        completed = self._load_event.wait(timeout=timeout)

        if not completed:
            # Timeout — switch to mock so callers are not blocked, but
            # let the background thread finish and restore real mode if
            # it eventually succeeds.
            with self._lock:
                if not self._bert_loaded:
                    logger.error(
                        "BERT loading timed out after %ds. "
                        "Switching to mock mode temporarily.", timeout
                    )
                    self.use_mock = True
                    self._bert_loaded = True
            return True

        return self._bert_loaded

    def _load_bert_worker(self, generation):
        """Background thread that downloads/loads BERT and commits the result."""
        try:
            logger.info("Loading BERT model: %s ...", self.model_name)
            try:
                tokenizer = BertTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                model = BertModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except Exception:
                logger.info("Local BERT not found, downloading...")
                tokenizer = BertTokenizer.from_pretrained(self.model_name)
                model = BertModel.from_pretrained(self.model_name)
            logger.info("BERT model loaded successfully.")

            with self._lock:
                # Only commit if our generation is still current.
                if generation != self._load_generation:
                    logger.info(
                        "BERT load finished but generation is stale (%d != %d), discarding.",
                        generation, self._load_generation,
                    )
                    return
                self.tokenizer = tokenizer
                self.model = model
                self.use_mock = False
                self._bert_loaded = True
        except Exception as e:
            logger.warning(
                "Failed to load BERT model '%s': %s. Falling back to mock mode.",
                self.model_name, e,
            )
            with self._lock:
                if generation == self._load_generation:
                    self.use_mock = True
                    self._bert_loaded = True
        finally:
            self._load_event.set()
            
    def _get_model_snapshot(self):
        """Return a consistent (use_mock, model, tokenizer) snapshot under lock."""
        with self._lock:
            return self.use_mock, self.model, self.tokenizer

    def encode(self, text):
        """Returns a vector representation of the text."""
        # Always ensure BERT is loaded before using (for non-mock mode)
        if not self.use_mock:
            self._ensure_bert_loaded(timeout=10)

        use_mock, model, tokenizer = self._get_model_snapshot()
        if use_mock or model is None or tokenizer is None:
            # Generate a deterministic pseudo-random vector based on text hash
            # This ensures same text gets same vector across process restarts.
            digest = hashlib.sha1(text.encode('utf-8')).hexdigest()
            seed = int(digest[:8], 16)
            rng = np.random.default_rng(seed)
            return rng.random(768) # Standard BERT size
        
        try:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # Use CLS token or mean pooling
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            return np.zeros(768)

    def encode_list(self, text_list):
        """Batch encode a list of texts."""
        if not self.use_mock:
            self._ensure_bert_loaded(timeout=10)

        use_mock, model, tokenizer = self._get_model_snapshot()
        if use_mock or model is None or tokenizer is None:
            return [self.encode(t) for t in text_list]

        embeddings = []
        batch_size = 16
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend([e for e in batch_emb])
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
        if norm_q == 0:
            return np.array([], dtype=int), np.array([])
        query_emb = query_emb / norm_q
        
        if candidate_embs is None or len(candidate_embs) == 0:
            return np.array([], dtype=int), np.array([])

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
        self.sim_model = SimilarityModel(use_mock=get_cfg(config, 'USE_MOCK_MODELS', False))
        
        # Load initial knowledge base for AC Automaton
        # In a real scenario, this would load from Neo4j or a file
        self._load_initial_data()
        
        # BERT Relation Extraction Setup
        self.relation_list = []
        self.relation_embs = None
        self._relation_embs_ready = False
        self._relation_embs_cache_path = None
        self._relation_embs_lock = threading.Lock()
        self._load_relations()

    def _load_relations(self):
        """Load relation names and defer embedding computation until needed."""
        rel_file = os.path.join(get_cfg(self.config, 'REGCN_DATA_DIR', ''), 'relation2id.txt')
        if not os.path.exists(rel_file):
            # Fallback path logic or hardcoded typical path
            rel_file = os.path.join(os.getcwd(), 'models', 'RE-GCN-master', 'data', '80STOCKS', 'relation2id.txt')
        
        cache_dir = os.path.join(get_cfg(self.config, 'BASE_PATH', os.getcwd()), 'data', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = None
        cache_path = None

        if os.path.exists(rel_file):
            logger.info(f"Loading relations from {rel_file}...")
            with open(rel_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        self.relation_list.append(parts[0])

            rel_bytes = ("\n".join(self.relation_list)).encode("utf-8")
            rel_hash = hashlib.sha1(rel_bytes).hexdigest()
            cache_key = f"{self.sim_model.model_name}_{len(self.relation_list)}_{rel_hash}"
            cache_path = os.path.join(cache_dir, f"relation_embs_{cache_key}.npz")
            self._relation_embs_cache_path = cache_path
            logger.info(f"Loaded {len(self.relation_list)} relations. Embeddings will be prepared lazily.")
        else:
            logger.warning("relation2id.txt not found. BERT Relation Extraction will be disabled.")

    def _ensure_relation_embeddings(self):
        """Compute or load cached relation embeddings on first real use."""
        if self._relation_embs_ready:
            return
        with self._relation_embs_lock:
            if self._relation_embs_ready:
                return
            if not self.relation_list:
                self.relation_embs = np.array([])
                self._relation_embs_ready = True
                return

            cache_path = self._relation_embs_cache_path
            if not self.sim_model.use_mock and cache_path and os.path.exists(cache_path):
                try:
                    logger.info(f"Loading cached relation embeddings from {cache_path}...")
                    loaded = np.load(cache_path)
                    self.relation_embs = loaded['embs']
                    self._relation_embs_ready = True
                    logger.info("Cached relation embeddings loaded.")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached embeddings: {e}. Recomputing...")

            logger.info(f"Computing BERT embeddings for {len(self.relation_list)} relations...")
            embs = self.sim_model.encode_list(self.relation_list)
            self.relation_embs = np.array(embs)
            self._relation_embs_ready = True
            if not self.sim_model.use_mock and cache_path:
                try:
                    np.savez_compressed(cache_path, embs=self.relation_embs)
                    logger.info(f"Relation embeddings cached to {cache_path}.")
                except Exception as e:
                    logger.warning(f"Failed to cache embeddings: {e}")
            logger.info("Relation embeddings computed.")
        
    def _load_initial_data(self):
        """Load entities from entity2id.txt and relations from relation2id.txt"""
        # 1. Load entities from entity2id.txt
        ent_file = os.path.join(get_cfg(self.config, 'REGCN_DATA_DIR', ''), 'entity2id.txt')
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

        rel_file = os.path.join(get_cfg(self.config, 'REGCN_DATA_DIR', ''), 'relation2id.txt')
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
        quadruple = {"h": "", "r": "", "t": "", "time": ""}
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
                
                # Check if this overlaps with existing matches to avoid duplication
                is_overlap = False
                for existing in enriched_matches:
                    if existing['start'] != -1 and not (m.end() - 1 < existing['start'] or m.start() > existing['end']):
                        is_overlap = True
                        break
                        
                if is_overlap:
                    continue
                    
                # If we haven't found a time yet, or this is more specific, take it
                if not quadruple['time']:
                    quadruple['time'] = word
                elif word not in quadruple['time']: 
                    # Append if different (e.g. 2018年 5月)
                    quadruple['time'] += word
                
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
            self._ensure_relation_embeddings()
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
                    second_score = scores[1] if len(scores) > 1 else 0.0
                    top_rel = self.relation_list[top_idx]

                    logger.info(f"BERT Top match: {top_rel} (score: {top_score:.4f}, margin: {top_score - second_score:.4f})")

                    # Stricter matching to avoid false positives:
                    # 1. High absolute threshold (0.85)
                    # 2. Top-1/Top-2 margin must be significant (>= 0.1) to avoid
                    #    ambiguous matches where multiple relations are equally similar
                    # 3. Blacklist of non-relation terms that should never be matched
                    RELATION_BLACKLIST = {"首席厨师", "厨师", "保洁", "保安"}
                    if remaining_text in RELATION_BLACKLIST:
                        logger.info(f"BERT match skipped: '{remaining_text}' is blacklisted")
                    elif top_score >= 0.85 and (top_score - second_score) >= 0.1:
                        quadruple['r'] = top_rel
                        enriched_matches.append({
                            "word": remaining_text,
                            "start": -1,
                            "end": -1,
                            "type": f"RELATION (BERT: {top_rel})"
                        })
                    else:
                        logger.info(
                            f"BERT match rejected: score={top_score:.4f} < 0.85 or "
                            f"margin={top_score - second_score:.4f} < 0.1"
                        )
        
        return {
            "original_text": text,
            "ac_matches": enriched_matches,
            "segmentation": seg_results,
            "structured_query": quadruple,
            "embedding_sample": embedding[:5].tolist() # Just show first 5 dims
        }
