import sys
import os
import torch
import numpy as np
import logging
from types import SimpleNamespace

# Configure Logging
logger = logging.getLogger(__name__)

class REGCNWrapper:
    def __init__(self, config):
        self.config = config
        self.regcn_base_dir = config['REGCN_BASE_DIR']
        self.dataset = config['REGCN_DATASET']
        self.model_path = config['REGCN_MODEL_PATH']
        self.data_dir = config['REGCN_DATA_DIR']
        self.params = config['REGCN_PARAMS']
        
        # Add REGCN to sys.path
        if self.regcn_base_dir not in sys.path:
            sys.path.append(self.regcn_base_dir)
            
        # Import REGCN modules here to avoid top-level import errors
        try:
            from src.rrgcn import RecurrentRGCN
            from rgcn import utils
            from rgcn.utils import build_sub_graph
            self.RecurrentRGCN = RecurrentRGCN
            self.utils = utils
            self.build_sub_graph = build_sub_graph
        except ImportError as e:
            logger.error(f"Failed to import REGCN modules: {e}")
            raise e

        # Initialize
        self.use_cuda = self.params['gpu'] >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device('cuda:' + str(self.params['gpu']))
        else:
            self.device = torch.device('cpu')
            
        self._load_data()
        self._load_model()
        
    def _load_data(self):
        logger.info(f"Loading REGCN data for dataset: {self.dataset}")
        
        # Load mappings
        def load_mapping(filename):
            mapping = {}
            rev_mapping = {}
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        name, id_ = parts[0], int(parts[1])
                        mapping[name] = id_
                        rev_mapping[id_] = name
            return mapping, rev_mapping

        self.entity2id, self.id2entity = load_mapping('entity2id.txt')
        self.relation2id, self.id2relation = load_mapping('relation2id.txt')
        
        # Time mapping is usually just sorted timestamps in train/test
        # We need to load time2id if it exists, or infer it
        if os.path.exists(os.path.join(self.data_dir, 'time2id.txt')):
            self.time2id, self.id2time = load_mapping('time2id.txt')
        else:
            # Fallback if no time2id (some datasets don't have it explicitly)
            self.time2id = {} 
            self.id2time = {}

        # Load graph data (train/valid/test) to build history
        # We use rgcn.utils.load_data which expects data in ../data usually
        # But we can monkey patch or just pass the right path logic
        # Actually rgcn.utils.load_data calls load_from_local which uses relative path "../data"
        # We might need to temporarily change CWD or use absolute paths if the utils supports it
        # Looking at utils.py: return knwlgrh.load_from_local("../data", dataset)
        # It's hardcoded to "../data". We need to handle this.
        
        # Alternative: We read the files directly using the logic from utils.py
        # But RecurrentRGCN expects specific data structures.
        # Let's try to load using utils but change the path or trick it.
        
        # Actually, since we added regcn_base_dir to sys.path, if we run from app root, 
        # utils.load_data will look for "../data" relative to where it thinks it is? 
        # No, relative paths are relative to CWD.
        # If CWD is D:\研究项目\问答系统模块, "../data" is D:\研究项目\data (which exists but empty?)
        # The data is at D:\研究项目\问答系统模块\models\RE-GCN-master\data
        
        # Let's manually load the triples as lists
        def read_triples(filename):
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                return []
            with open(path, 'r') as f:
                data = []
                for line in f:
                    data.append(list(map(int, line.strip().split())))
            return np.array(data)

        self.train_data = read_triples('train.txt')
        self.valid_data = read_triples('valid.txt')
        self.test_data = read_triples('test.txt')
        
        # Combine all data to build full history
        all_data = []
        if len(self.train_data) > 0: all_data.append(self.train_data)
        if len(self.valid_data) > 0: all_data.append(self.valid_data)
        if len(self.test_data) > 0: all_data.append(self.test_data)
        self.all_data = np.concatenate(all_data)
        
        # Sort by time
        self.all_data = self.all_data[self.all_data[:, 3].argsort()]

        self.kg_index_time = {}
        for s, r, o, t in self.all_data:
            key = (int(s), int(r), int(t))
            if key not in self.kg_index_time:
                self.kg_index_time[key] = set()
            self.kg_index_time[key].add(int(o))
        
        self.num_nodes = len(self.entity2id)
        self.num_rels = len(self.relation2id)
        
        # Pre-process snapshots for history retrieval
        self.snapshots = self.utils.split_by_time(self.all_data)
        # Map time_id to snapshot index
        self.time_to_snap_idx = {}
        for idx, snap in enumerate(self.snapshots):
            # Assumes all triples in a snapshot have the same time
            # Check the first triple's time
            # Note: split_by_time returns [s, r, o] only (stripped time), 
            # but we need to know which time corresponds to which snapshot.
            # Let's rewrite split_by_time logic slightly or map it back.
            
            # Actually, split_by_time in utils.py relies on the data being sorted by time.
            # And it returns list of numpy arrays of (s, r, o).
            # We can reconstruct the time mapping.
            # The all_data is sorted.
            start_idx = 0
            for i, snap in enumerate(self.snapshots):
                # Get the time from the original data corresponding to this snapshot
                # Length of snap
                length = len(snap)
                # The time of this block in all_data
                t = self.all_data[start_idx][3]
                self.time_to_snap_idx[t] = i
                start_idx += length
                
        logger.info(f"Data loaded. Nodes: {self.num_nodes}, Rels: {self.num_rels}, Snapshots: {len(self.snapshots)}")

    def _load_model(self):
        logger.info("Loading REGCN model...")
        p = self.params
        
        self.model = self.RecurrentRGCN(
            p['decoder'],
            p['encoder'],
            self.num_nodes,
            self.num_rels,
            0, # num_static_rels
            0, # num_words
            p['n_hidden'],
            p['opn'],
            sequence_len=p['train_history_len'],
            num_bases=p['n_bases'],
            num_basis=p['n_basis'],
            num_hidden_layers=p['n_layers'],
            dropout=p['dropout'],
            self_loop=p['self_loop'],
            skip_connect=p['skip_connect'],
            layer_norm=p['layer_norm'],
            input_dropout=p['input_dropout'],
            hidden_dropout=p['hidden_dropout'],
            feat_dropout=p['feat_dropout'],
            aggregation=p['aggregation'],
            weight=p['weight'],
            discount=p['discount'],
            angle=p['angle'],
            use_static=p['add_static_graph'],
            entity_prediction=p['entity_prediction'],
            relation_prediction=p['relation_prediction'],
            use_cuda=self.use_cuda,
            gpu=p['gpu'] if self.use_cuda else 'cpu',
            analysis=p['run_analysis']
        )
        
        if self.use_cuda:
            self.model.cuda()
            checkpoint = torch.load(self.model_path, map_location=self.device)
        else:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        logger.info("REGCN model loaded successfully.")

    def predict(self, head_name, relation_name, time_str=None, top_k=5):
        """
        Predict tail entities given (head, relation, time).
        """
        # 1. Map inputs to IDs
        h_id = self.entity2id.get(head_name)
        r_id = self.relation2id.get(relation_name)
        
        if h_id is None:
            return [{"name": f"Unknown entity: {head_name}", "score": 0.0, "source": "Input Error"}]
        if r_id is None:
            return [{"name": f"Unknown relation: {relation_name}", "score": 0.0, "source": "Input Error"}]
            
        # Time handling
        # If time_str is provided, map to ID. 
        # If no time provided, use the latest time in dataset? Or specific logic?
        # The CSV used 20250101 -> ID 0.
        # If input is '20250102', we need to find its ID.
        t_id = self._resolve_time_id(time_str)
        
        if t_id is None:
            # Default to latest time or 0 if user didn't specify
            # But wait, 0 is the START. We probably want to predict for the future?
            # Or if it's a historical query, use that time.
            # If unknown time, we can't easily position it in history.
            # Let's fallback to the last timestamp in data if not found.
            t_id = self.all_data[-1][3]
            logger.warning(f"Time {time_str} not found or not provided. Using last available time ID: {t_id}")

        # 1.5 EXACT MATCH RETRIEVAL (Moved to separate method)
        # We do NOT return DB results here anymore, to keep 'predict' pure for reasoning.
        # The app should call get_fact() separately to populate the KB section.
            
        # 2. Construct History
        # We need the sequence of graphs ending BEFORE t_id.
        # Find the snapshot index for t_id
        current_snap_idx = self.time_to_snap_idx.get(t_id)
        
        if current_snap_idx is None:
             # If t_id is future (not in data), use the last available snapshots
             current_snap_idx = len(self.snapshots)
        
        history_len = self.params['train_history_len']
        if current_snap_idx < history_len:
            current_snap_idx = history_len
        start_idx = max(0, current_snap_idx - history_len)
        input_snaps = self.snapshots[start_idx : current_snap_idx]
        
        # If history is too short, pad? Or just use what we have (RGCN might handle it or need padding)
        # RecurrentRGCN iterates over the list. Short list is fine.
        
        if not input_snaps:
            return [{"name": "No history data available for this time.", "score": 0.0, "source": "System Error"}]

        history_glist = [
            self.build_sub_graph(self.num_nodes, self.num_rels, snap, self.use_cuda, self.params['gpu'] if self.use_cuda else 'cpu') 
            for snap in input_snaps
        ]
        
        # 3. Prepare Test Triplet
        # (h, r, ?) -> we represent this as (h, r, 0) just to fit the shape
        test_triples = torch.LongTensor([[h_id, r_id, 0]])
        if self.use_cuda:
            test_triples = test_triples.cuda()
            
        # 4. Predict
        # predict(test_graph, num_rels, static_graph, test_triplets, use_cuda)
        # static_graph is None as per config
        _, scores, _ = self.model.predict(history_glist, self.num_rels, None, test_triples, self.use_cuda)
        
        # scores is [1, num_ents]
        scores = scores[0].cpu().numpy()
        
        # 5. Get Top K
        # scores is raw logits (can be negative or positive, unbounded)
        # Apply Softmax to get probabilities that sum to 1 across all entities
        # This makes more sense for "Who is the Chairman?" (Mutually exclusive-ish)
        # But RE-GCN is multi-label. Sigmoid is standard for KGE.
        # However, if scores are very large (e.g. > 10), sigmoid(score) -> 1.0
        # If model is overconfident, all top K will be 1.0.
        
        # Let's use Softmax over the Top K (or Top N) to force distribution if user wants "probability"
        # OR just show raw scores? No, user wants probability.
        
        # Standard approach in Link Prediction: Sigmoid.
        # If all are 1.0, it means the model is very confident about multiple entities.
        # This might happen if the relation is 1-to-N or model is not calibrated.
        
        # To fix the user's confusion "Why are there so many 1.0?", we can:
        # 1. Show raw scores? No.
        # 2. Use Softmax across the whole vocabulary (or at least top K) to make them compete.
        
        scores_tensor = torch.from_numpy(scores)
        probs = torch.softmax(scores_tensor, dim=0).numpy()
        
        top_indices = probs.argsort()[-top_k:][::-1]
        
        model_results = []
        for idx in top_indices:
            prob = float(probs[idx])
            # Filter low probability results
            if prob < 0.05:
                continue
            
            ent_name = self.id2entity.get(idx, f"Unknown ID {idx}")
            # Format score to 2 decimal places
            model_results.append({"name": ent_name, "score": round(prob, 2), "source": "Model Prediction"})
            
        return model_results
        
    def _resolve_time_id(self, time_str):
        if not time_str:
            return None
        if time_str in self.time2id:
            return self.time2id[time_str]
        digits = ''.join(ch for ch in str(time_str) if ch.isdigit())
        if len(digits) >= 8:
            key = digits[:8]
            return self.time2id.get(key)
        if len(digits) >= 6:
            prefix = digits[:6]
        elif len(digits) >= 4:
            prefix = digits[:4]
        else:
            return None
        candidates = [k for k in self.time2id.keys() if str(k).startswith(prefix)]
        if not candidates:
            return None
        best = max(candidates)
        return self.time2id.get(best)

    def get_fact(self, head_name, relation_name, time_str=None):
        """
        Retrieve exact fact from internal KG index.
        Returns list of strings (entity names).
        """
        h_id = self.entity2id.get(head_name)
        r_id = self.relation2id.get(relation_name)
        
        if h_id is None or r_id is None:
            return []
            
        t_id = self._resolve_time_id(time_str)
            
        if t_id is None:
            return []
             
        if (h_id, r_id, t_id) in self.kg_index_time:
            tail_ids = self.kg_index_time[(h_id, r_id, t_id)]
            return [self.id2entity.get(tid, str(tid)) for tid in tail_ids]
            
        return []
