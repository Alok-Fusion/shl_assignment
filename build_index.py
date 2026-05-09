"""
One-time script: Build FAISS index from catalog.json and save to faiss_index/.
Run locally once, then commit faiss_index/ to the repo.

Usage:
    python build_index.py
"""

import json
import os
import time
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Key abbreviation map ──────────────────────────────────────────────
KEY_ABBREV = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def build_test_type(keys: list[str]) -> str:
    """Comma-joined first-letter abbreviations from keys."""
    return ",".join(KEY_ABBREV.get(k, "?") for k in keys)


def build_embed_text(item: dict) -> str:
    """Construct the embedding text for a catalog item."""
    name = item["name"]
    description = item.get("description", "")
    keys = ", ".join(item.get("keys", []))
    job_levels = ", ".join(item.get("job_levels", []))
    languages = ", ".join(item.get("languages", [])[:5])
    return (
        f"{name}. {description}. Keys: {keys}. "
        f"Job levels: {job_levels}. Languages: {languages}"
    )


def embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts using Google text-embedding-004."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type=task_type,
    )
    return result["embedding"]


def main():
    # 1. Load catalog
    with open("catalog.json", "r", encoding="utf-8") as f:
        catalog = json.loads(f.read(), strict=False)

    print(f"Loaded {len(catalog)} catalog items.")

    # Fix known newline issue in names
    for item in catalog:
        item["name"] = item["name"].replace("\n", " ").strip()

    # 2. Build embed texts
    embed_texts = [build_embed_text(item) for item in catalog]

    # 3. Embed in batches of 10 with rate-limit handling
    #    Free tier: 100 embed requests/min/model (each item in batch counts)
    all_embeddings = []
    batch_size = 10
    total_batches = (len(embed_texts) + batch_size - 1) // batch_size
    for i in range(0, len(embed_texts), batch_size):
        batch = embed_texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} items)...")

        # Retry with exponential backoff on rate-limit errors
        max_retries = 5
        for attempt in range(max_retries):
            try:
                embeddings = embed_batch(batch)
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait = min(30 * (2 ** attempt), 120)
                    print(f"  Rate limited. Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait)
                else:
                    raise

        # Delay between batches to stay well under 100 requests/min
        if i + batch_size < len(embed_texts):
            time.sleep(8)

    # 4. Build FAISS IndexFlatIP (inner product → normalize vectors first)
    vectors = np.array(all_embeddings, dtype="float32")
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"FAISS index built: {index.ntotal} vectors, dimension {dim}")

    # 5. Build metadata
    metadata = []
    for item in catalog:
        metadata.append({
            "name": item["name"],
            "link": item["link"],
            "keys": item.get("keys", []),
            "test_type": build_test_type(item.get("keys", [])),
            "description": item.get("description", ""),
            "duration": item.get("duration", ""),
            "languages": item.get("languages", []),
            "job_levels": item.get("job_levels", []),
            "remote": item.get("remote", "no"),
            "adaptive": item.get("adaptive", "no"),
        })

    # 6. Save
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")
    with open("faiss_index/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Saved faiss_index/index.faiss and faiss_index/metadata.json")
    print("Done! Commit faiss_index/ to your repo.")


if __name__ == "__main__":
    main()
