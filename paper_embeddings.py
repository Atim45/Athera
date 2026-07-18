import os
import faiss
import numpy as np
import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_FILE = "paper_embeddings.npy"
INDEX_FILE = "paper_faiss.index"


def load_dataset_data():

    print("\nLoading Dataset...")
    dataset = load_dataset(
        "CShorten/ML-ArXiv-Papers",
        split="train"
    )
    df = pd.DataFrame(dataset)
    df = df[["title", "abstract"]]
    df = df.head(15000)
    df["paper_text"] = (
        df["title"] + " " + df["abstract"]
    )
    df["paper_text"] = (
        df["paper_text"]
        .fillna("")
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )
    print(f"Loaded {len(df)} papers.\n")
    return df


def load_embedding_model():

    print("Loading Sentence Transformer...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.\n")
    return model


def load_or_create_embeddings(df, model):

    if os.path.exists(EMBEDDING_FILE):
        print("Loading saved embeddings...\n")
        embeddings = np.load(EMBEDDING_FILE)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        print(f"Loaded embeddings shape={embeddings.shape}, dtype={embeddings.dtype}\n")

    else:
        print("Generating embeddings...\n")
        embeddings = model.encode(
            df["paper_text"].tolist(),
            batch_size=32,
            show_progress_bar=True
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(EMBEDDING_FILE, embeddings)
        print("Embeddings saved successfully.\n")
    return embeddings


def load_or_create_index(embeddings):

    if os.path.exists(INDEX_FILE):
        print("Loading existing FAISS index...\n")
        index = faiss.read_index(INDEX_FILE)

    else:
        print("Creating FAISS index...\n")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)
        print("FAISS index saved successfully.\n")
    return index

def initialize():

    df = load_dataset_data()
    model = load_embedding_model()
    embeddings = load_or_create_embeddings(df, model)
    index = load_or_create_index(embeddings)
    return df, model, embeddings, index