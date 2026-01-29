import os
import sys
import logging
from types import SimpleNamespace

import numpy as np
import torch
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class TiRGNWrapper:
    def __init__(self, config):
        self.config = config
        self.base_dir = config["TIRGN_BASE_DIR"]
        self.dataset = config["TIRGN_DATASET"]
        self.data_dir = config["TIRGN_DATA_DIR"]
        self.history_dir = config["TIRGN_HISTORY_DIR"]
        self.model_path = config["TIRGN_MODEL_PATH"]
        self.params = config["TIRGN_PARAMS"]

        if self.base_dir not in sys.path:
            sys.path.insert(0, self.base_dir)

        for name in list(sys.modules.keys()):
            if name == "rgcn" or name.startswith("rgcn.") or name == "src" or name.startswith("src."):
                del sys.modules[name]

        try:
            from rgcn import utils
            from rgcn.utils import build_sub_graph
            from rgcn.knowledge_graph import _read_triplets_as_list
            from src.rrgcn import RecurrentRGCN
            self.utils = utils
            self.build_sub_graph = build_sub_graph
            self._read_triplets_as_list = _read_triplets_as_list
            self.RecurrentRGCN = RecurrentRGCN
        except Exception as e:
            logger.error(f"Failed to import TiRGN modules: {e}")
            raise

        gpu = int(self.params.get("gpu", -1))
        self.use_cuda = gpu >= 0 and torch.cuda.is_available()
        self.gpu = gpu if self.use_cuda else "cpu"
        self.device = torch.device(f"cuda:{gpu}") if self.use_cuda else torch.device("cpu")

        self._tail_history_cache = {}
        self._rel_history_cache = {}

        self._load_mappings()
        self._load_data()
        self._load_static_graph()
        self._load_model()

    def _load_mappings(self):
        def load_mapping(filename):
            mapping = {}
            rev_mapping = {}
            path = os.path.join(self.data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.rstrip("\n")
                    parts = raw.split("\t")
                    if len(parts) != 2:
                        continue
                    name = parts[0].strip()
                    if not name:
                        continue
                    id_ = int(parts[1].strip())
                    mapping[name] = id_
                    rev_mapping[id_] = name
            return mapping, rev_mapping

        self.entity2id, self.id2entity = load_mapping("entity2id.txt")
        self.relation2id, self.id2relation = load_mapping("relation2id.txt")
        self.time2id, self.id2time = load_mapping("time2id.txt")

        self.num_nodes = len(self.entity2id)
        self.num_rels = len(self.relation2id)

    def _read_triples(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return np.zeros((0, 4), dtype=np.int64)
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                data.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
        return np.asarray(data, dtype=np.int64)

    def _load_data(self):
        train = self._read_triples("train.txt")
        valid = self._read_triples("valid.txt")
        test = self._read_triples("test.txt")
        parts = []
        if len(train):
            parts.append(train)
        if len(valid):
            parts.append(valid)
        if len(test):
            parts.append(test)
        if not parts:
            raise RuntimeError("TiRGN dataset is empty (train/valid/test not found).")
        self.all_data = np.concatenate(parts, axis=0)
        self.all_data = self.all_data[self.all_data[:, 3].argsort()]

        out = self.utils.split_by_time(self.all_data)
        if isinstance(out, tuple) or isinstance(out, list) and len(out) == 2 and isinstance(out[0], list):
            try:
                self.snapshots = out[0]
                self.times = np.asarray(out[1], dtype=np.int64)
            except Exception:
                self.snapshots = out[0]
                self.times = np.unique(self.all_data[:, 3].astype(np.int64))
        else:
            self.snapshots = out
            self.times = np.unique(self.all_data[:, 3].astype(np.int64))
        self.times = np.asarray(sorted(set(map(int, self.times.tolist()))), dtype=np.int64)
        self.time_to_snap_idx = {}
        start_idx = 0
        for i, snap in enumerate(self.snapshots):
            t = int(self.all_data[start_idx][3])
            self.time_to_snap_idx[t] = i
            start_idx += len(snap)

    def _load_static_graph(self):
        if not self.params.get("use_static", True):
            self.static_graph = None
            self.num_static_rels = 0
            self.num_words = 0
            return

        path = os.path.join(self.data_dir, "e-w-graph.txt")
        if not os.path.exists(path):
            self.static_graph = None
            self.num_static_rels = 0
            self.num_words = 0
            return

        static_triples = np.asarray(self._read_triplets_as_list(path, {}, {}, load_time=False), dtype=np.int64)
        self.num_static_rels = int(len(np.unique(static_triples[:, 1])))
        self.num_words = int(len(np.unique(static_triples[:, 2])))
        static_triples[:, 2] = static_triples[:, 2] + self.num_nodes
        static_node_count = self.num_nodes + self.num_words
        self.static_graph = self.build_sub_graph(static_node_count, self.num_static_rels, static_triples, self.use_cuda, self.gpu)

    def _build_model(self, entity_prediction, relation_prediction, layer_norm, use_static):
        p = self.params
        num_times = int(p.get("num_times", len(self.snapshots)))
        time_interval = int(p.get("time_interval", 1))
        return self.RecurrentRGCN(
            p["decoder"],
            p["encoder"],
            self.num_nodes,
            self.num_rels,
            self.num_static_rels if use_static else 0,
            self.num_words if use_static else 0,
            num_times,
            time_interval,
            int(p.get("n_hidden", 200)),
            p.get("opn", "sub"),
            float(p.get("history_rate", 0.3)),
            sequence_len=int(p.get("train_history_len", 9)),
            num_bases=int(p.get("n_bases", 100)),
            num_basis=int(p.get("n_basis", 100)),
            num_hidden_layers=int(p.get("n_layers", 2)),
            dropout=float(p.get("dropout", 0.2)),
            self_loop=bool(p.get("self_loop", True)),
            skip_connect=bool(p.get("skip_connect", False)),
            layer_norm=layer_norm,
            input_dropout=float(p.get("input_dropout", 0.2)),
            hidden_dropout=float(p.get("hidden_dropout", 0.2)),
            feat_dropout=float(p.get("feat_dropout", 0.2)),
            aggregation=p.get("aggregation", "none"),
            weight=float(p.get("weight", 0.5)),
            discount=float(p.get("discount", 1.0)),
            angle=int(p.get("angle", 14)),
            use_static=use_static,
            entity_prediction=entity_prediction,
            relation_prediction=relation_prediction,
            use_cuda=self.use_cuda,
            gpu=self.gpu,
            analysis=bool(p.get("run_analysis", False)),
        )

    def _load_model(self):
        logger.info("Loading TiRGN model...")
        tried = []
        candidates = [
            (True, True, True, True),
            (True, True, False, True),
            (True, False, True, True),
            (True, False, False, True),
            (True, True, True, False),
            (True, True, False, False),
            (True, False, True, False),
            (True, False, False, False),
        ]

        checkpoint = torch.load(self.model_path, map_location=self.device)
        try:
            w = checkpoint["state_dict"].get("linear_0.weight")
            if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                self.params["num_times"] = int(w.shape[1])
        except Exception:
            pass
        last_error = None
        for entity_pred, rel_pred, layer_norm, use_static in candidates:
            tried.append((entity_pred, rel_pred, layer_norm, use_static))
            try:
                model = self._build_model(entity_pred, rel_pred, layer_norm, use_static)
                if self.use_cuda:
                    model.cuda()
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                self.model = model
                self.model_flags = SimpleNamespace(
                    entity_prediction=entity_pred,
                    relation_prediction=rel_pred,
                    layer_norm=layer_norm,
                    use_static=use_static,
                )
                logger.info(
                    "TiRGN model loaded successfully. "
                    f"entity_prediction={entity_pred}, relation_prediction={rel_pred}, "
                    f"layer_norm={layer_norm}, use_static={use_static}"
                )
                return
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"Failed to load TiRGN model checkpoint with candidates {tried}: {last_error}")

    def _resolve_time_id(self, time_str):
        if not time_str:
            return None
        if time_str in self.time2id:
            return int(self.time2id[time_str])
        digits = "".join(ch for ch in str(time_str) if ch.isdigit())
        if len(digits) >= 8:
            key = digits[:8]
            if key in self.time2id:
                return int(self.time2id[key])
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
        return int(self.time2id[best])

    def _load_history_npz(self, kind, time_id):
        cache = self._tail_history_cache if kind == "tail" else self._rel_history_cache
        if time_id in cache:
            return cache[time_id]
        filename = f"{kind}_history_{time_id}.npz"
        path = os.path.join(self.history_dir, filename)
        mat = sp.load_npz(path)
        if len(cache) >= 8:
            cache.clear()
        cache[time_id] = mat
        return mat

    def _build_history_vocab(self, test_triplets, time_id):
        histroy_data = test_triplets
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]].clone()
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + self.num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data]).cpu().numpy()

        all_tail_seq = self._load_history_npz("tail", time_id)
        seq_idx = histroy_data[:, 0] * self.num_rels * 2 + histroy_data[:, 1]
        tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
        one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

        all_rel_seq = self._load_history_npz("rel", time_id)
        rel_seq_idx = histroy_data[:, 0] * self.num_nodes + histroy_data[:, 2]
        rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
        one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)

        if self.use_cuda:
            one_hot_tail_seq = one_hot_tail_seq.cuda(self.gpu)
            one_hot_rel_seq = one_hot_rel_seq.cuda(self.gpu)
        return one_hot_tail_seq, one_hot_rel_seq

    def predict_tail(self, head_name, relation_name, time_str=None, top_k=5):
        head_name = (head_name or "").strip()
        relation_name = (relation_name or "").strip()
        h_id = self.entity2id.get(head_name)
        r_id = self.relation2id.get(relation_name)
        if h_id is None:
            return [{"name": f"Unknown entity: {head_name}", "score": 0.0, "source": "Input Error"}]
        if r_id is None:
            return [{"name": f"Unknown relation: {relation_name}", "score": 0.0, "source": "Input Error"}]

        t_id = self._resolve_time_id(time_str)
        if t_id is None:
            t_id = int(self.times.max())
        snap_idx = self.time_to_snap_idx.get(t_id, None)
        history_len = int(self.params.get("train_history_len", 9))

        if snap_idx is None:
            end_idx = len(self.snapshots)
        else:
            end_idx = snap_idx
        start_idx = max(0, end_idx - history_len)
        history_snaps = self.snapshots[start_idx:end_idx]
        if not history_snaps:
            history_snaps = self.snapshots[max(0, len(self.snapshots) - history_len):]
        if not history_snaps:
            return [{"name": "No history data available.", "score": 0.0, "source": "System Error"}]

        history_glist = [
            self.build_sub_graph(self.num_nodes, self.num_rels, snap, self.use_cuda, self.gpu)
            for snap in history_snaps
        ]

        test_triplets = torch.LongTensor([[int(h_id), int(r_id), 0, int(t_id)]])
        if self.use_cuda:
            test_triplets = test_triplets.cuda(self.gpu)

        one_hot_tail_seq, one_hot_rel_seq = self._build_history_vocab(test_triplets, int(t_id))

        all_triples, score_log, _ = self.model.predict(
            history_glist,
            self.num_rels,
            self.static_graph if getattr(self, "model_flags", SimpleNamespace(use_static=False)).use_static else None,
            test_triplets,
            one_hot_tail_seq,
            one_hot_rel_seq,
            self.use_cuda,
        )

        probs = torch.exp(score_log[0]).detach().cpu().numpy()
        top_indices = probs.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            prob = float(probs[idx])
            results.append({"name": str(self.id2entity.get(int(idx), str(idx))).strip(), "score": prob, "source": "TiRGN"})
        return results

