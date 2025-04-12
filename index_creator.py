import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

# === CONFIG ===
BOOK_PATH = "book_compilation.txt"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # good balance of speed and quality
INDEX_PATH = "faiss_index.index"
CHUNKS_PATH = "chunks.pkl"

# === STEP 1: Load the book ===
with open(BOOK_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# === STEP 2: Split into overlapping chunks ===
def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

# === STEP 3: Embed chunks ===
print(f"Embedding {len(chunks)} chunks...")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(chunks, show_progress_bar=True)

# === STEP 4: Create FAISS index ===
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# === STEP 5: Save index and chunks ===
faiss.write_index(index, INDEX_PATH)
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("Index and chunk list saved!")