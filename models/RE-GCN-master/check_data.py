
import os
import numpy as np

data_dir = "data/TKG_SAMPLE"
files = ["entity2id.txt", "train.txt", "valid.txt", "test.txt", "e-w-graph.txt"]

def read_id_file(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ids.append(int(parts[1]))
    return ids

def read_triple_file(path, is_ew=False):
    srcs = []
    rels = []
    dsts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                srcs.append(int(parts[0]))
                rels.append(int(parts[1]))
                dsts.append(int(parts[2]))
    return srcs, rels, dsts

print("Checking entity2id.txt...")
entity_ids = read_id_file(os.path.join(data_dir, "entity2id.txt"))
print(f"Number of entities in entity2id.txt: {len(entity_ids)}")
if entity_ids:
    print(f"Max entity ID: {max(entity_ids)}")
    print(f"Min entity ID: {min(entity_ids)}")

print("\nChecking train.txt...")
train_src, train_rel, train_dst = read_triple_file(os.path.join(data_dir, "train.txt"))
print(f"Max train src: {max(train_src)}, Max train dst: {max(train_dst)}")
print(f"Max train rel: {max(train_rel)}")

print("\nChecking valid.txt...")
valid_src, valid_rel, valid_dst = read_triple_file(os.path.join(data_dir, "valid.txt"))
print(f"Max valid src: {max(valid_src)}, Max valid dst: {max(valid_dst)}")

print("\nChecking test.txt...")
test_src, test_rel, test_dst = read_triple_file(os.path.join(data_dir, "test.txt"))
print(f"Max test src: {max(test_src)}, Max test dst: {max(test_dst)}")

print("\nChecking e-w-graph.txt...")
ew_src, ew_rel, ew_dst = read_triple_file(os.path.join(data_dir, "e-w-graph.txt"))
print(f"Max ew src (entity): {max(ew_src)}")
print(f"Max ew dst (word): {max(ew_dst)}")
print(f"Max ew rel: {max(ew_rel)}")

unique_words = sorted(list(set(ew_dst)))
print(f"Number of unique words: {len(unique_words)}")
if len(unique_words) != max(ew_dst) + 1:
    print("WARNING: Gaps in word IDs detected!")
    print(f"Expected {max(ew_dst) + 1} words, but found {len(unique_words)}")

max_entity_id_used = max(
    max(train_src) if train_src else 0,
    max(train_dst) if train_dst else 0,
    max(valid_src) if valid_src else 0,
    max(valid_dst) if valid_dst else 0,
    max(test_src) if test_src else 0,
    max(test_dst) if test_dst else 0,
    max(ew_src) if ew_src else 0
)

print(f"\nMax entity ID used in triples: {max_entity_id_used}")
