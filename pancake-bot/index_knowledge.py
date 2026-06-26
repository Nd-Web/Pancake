"""
index_knowledge.py — Knowledge Base Indexer

Reads all .txt files from the knowledge/ folder, splits them into
overlapping chunks, generates embeddings using sentence-transformers,
and stores everything in a persistent ChromaDB vector database.

Run this script once before starting the webhook server so that the
knowledge base is populated and ready for similarity searches.
"""

import os
import glob
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory containing the raw knowledge .txt files
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")

# Path where ChromaDB will persist its data to disk
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Name of the ChromaDB collection that holds our knowledge chunks
COLLECTION_NAME = "pancake_knowledge"

# Chunking parameters — how we split each text file into pieces
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks for context continuity

# The sentence-transformers model used to generate embeddings.
# all-MiniLM-L6-v2 is a good balance of speed and quality for retrieval.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_text_files(directory: str) -> dict[str, str]:
    """
    Load every .txt file in the given directory.

    Returns:
        A dictionary mapping filename -> file contents.
    """
    files = {}
    # Use glob to find all .txt files in the knowledge directory
    pattern = os.path.join(directory, "*.txt")
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            files[filename] = f.read()
    return files


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a single text string into overlapping chunks.

    Overlap ensures that information at chunk boundaries isn't lost.
    For example, with chunk_size=500 and overlap=50, the last 50
    characters of chunk N become the first 50 characters of chunk N+1.

    Args:
        text:       The full text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    # Walk through the text in steps of (chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # skip empty chunks that might arise from whitespace
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_knowledge():
    """
    Main indexing routine:
      1. Load all .txt knowledge files.
      2. Split each file into overlapping chunks.
      3. Generate embeddings and store in ChromaDB.
    """
    # Load environment variables (not strictly needed here but keeps
    # things consistent if we add config via .env later)
    load_dotenv()

    # ------------------------------------------------------------------
    # Step 1 — Load raw text files
    # ------------------------------------------------------------------
    print(f"📂 Loading knowledge files from: {KNOWLEDGE_DIR}")
    text_files = load_text_files(KNOWLEDGE_DIR)

    if not text_files:
        print("⚠️  No .txt files found in the knowledge/ directory. Nothing to index.")
        return

    print(f"   Found {len(text_files)} file(s): {', '.join(text_files.keys())}")

    # ------------------------------------------------------------------
    # Step 2 — Initialize ChromaDB
    # ------------------------------------------------------------------
    # PersistentClient saves data to disk so we don't re-index every restart.
    print(f"\n🗄️  Initializing ChromaDB at: {CHROMA_DB_DIR}")
    client = PersistentClient(path=CHROMA_DB_DIR)

    # Get or create the collection. If it already exists from a previous
    # run, we delete it first so we get a fresh, clean index.
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"   Existing collection '{COLLECTION_NAME}' found — deleting for a fresh index.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
    )
    print(f"   Collection '{COLLECTION_NAME}' created successfully.")

    # ------------------------------------------------------------------
    # Step 3 — Load the embedding model
    # ------------------------------------------------------------------
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("   Model loaded.")

    # ------------------------------------------------------------------
    # Step 4 — Chunk, embed, and store each file
    # ------------------------------------------------------------------
    all_ids = []
    all_documents = []
    all_embeddings = []
    all_metadatas = []
    chunk_counter = 0

    for filename, content in text_files.items():
        # Split the file content into chunks
        chunks = chunk_text(content)
        print(f"\n   📄 {filename}: {len(chunks)} chunk(s) generated")

        for i, chunk in enumerate(chunks):
            # Generate the embedding vector for this chunk
            embedding = embedding_model.encode(chunk).tolist()

            # Create a unique ID for each chunk
            chunk_id = f"{filename}_chunk_{i}"

            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_embeddings.append(embedding)
            # Store the source filename as metadata so we can trace back
            all_metadatas.append({"source": filename})
            chunk_counter += 1

    # Batch-insert all chunks into ChromaDB in one call for efficiency
    collection.add(
        ids=all_ids,
        documents=all_documents,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    print(f"\n✅ Indexing complete! {chunk_counter} chunk(s) stored in collection '{COLLECTION_NAME}'.")
    print(f"   Database persisted at: {CHROMA_DB_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    index_knowledge()
