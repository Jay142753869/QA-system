
import os
import numpy as np
import shutil

data_dir = "data/TKG_SAMPLE"
file_path = os.path.join(data_dir, "e-w-graph.txt")
backup_path = os.path.join(data_dir, "e-w-graph.txt.bak")

# Read the file
print(f"Reading {file_path}...")
triples = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            triples.append([int(parts[0]), int(parts[1]), int(parts[2])])

triples = np.array(triples)
print(f"Original shape: {triples.shape}")

# Check word IDs (column 2)
words = triples[:, 2]
unique_words = np.unique(words)
print(f"Number of unique words: {len(unique_words)}")
print(f"Max word ID: {max(words)}")
print(f"Min word ID: {min(words)}")

if len(unique_words) == max(words) + 1:
    print("Word IDs are already contiguous. No fix needed.")
else:
    print("Word IDs are NOT contiguous. Remapping...")
    
    # Backup original file
    if not os.path.exists(backup_path):
        print(f"Backing up to {backup_path}")
        shutil.copy(file_path, backup_path)
    else:
        print(f"Backup already exists at {backup_path}")

    # Remap
    # unique_words are sorted.
    # Map old_id -> new_id (index in unique_words)
    word_map = {old_id: new_id for new_id, old_id in enumerate(unique_words)}
    
    new_words = np.array([word_map[w] for w in words])
    triples[:, 2] = new_words
    
    print(f"New max word ID: {max(triples[:, 2])}")
    
    # Write back
    print(f"Writing fixed data to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        for row in triples:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
            
    print("Done.")
