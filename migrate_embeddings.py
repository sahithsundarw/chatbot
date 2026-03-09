"""
One-time migration: re-embeds the existing chroma_db using Google AI Embeddings.
Run this once locally, then commit and push the new chroma_db/.
"""
import gc
import os
import shutil
import subprocess
import sys
import time
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv(override=True)

OLD_DB = os.path.abspath("./chroma_db")
NEW_DB = os.path.abspath("./chroma_db_new")

print("Step 1: Reading all documents from existing chroma_db...")
client = chromadb.PersistentClient(path=OLD_DB)
collection = client.list_collections()[0]
existing = collection.get(include=["documents", "metadatas"])

docs = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(existing["documents"], existing["metadatas"])
]
print(f"  -> Found {len(docs)} documents")

# Close the client to release file locks before any directory operations
del collection
del client

print("\nStep 2: Creating new chroma_db with Google AI Embeddings...")
print("  (This will make API calls — may take a few minutes)")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

if os.path.exists(NEW_DB):
    gc.collect()
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", NEW_DB], check=True)
    else:
        shutil.rmtree(NEW_DB)

# Embed in batches of 100, with 65s delay to respect free-tier rate limits (1 RPM)
BATCH = 100
total_batches = (len(docs) + BATCH - 1) // BATCH
new_db = None
for i in range(0, len(docs), BATCH):
    batch_num = i // BATCH + 1
    batch = docs[i:i + BATCH]
    print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} docs)...")
    if new_db is None:
        new_db = Chroma.from_documents(batch, embedding_model, persist_directory=NEW_DB)
    else:
        new_db.add_documents(batch)
    if batch_num < total_batches:
        print(f"    Waiting 65s to stay under 1 RPM free-tier limit...")
        time.sleep(65)

# Close new_db before moving directories
del new_db

print("\nStep 3: Replacing old chroma_db with new one...")
gc.collect()
if sys.platform == "win32":
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", OLD_DB], check=True)
else:
    shutil.rmtree(OLD_DB)
shutil.move(NEW_DB, OLD_DB)

print("\nDone! chroma_db/ now uses Google AI Embeddings.")
print("Next: git add chroma_db/ && git commit -m 'Rebuild vector store with Google AI embeddings' && git push")
