"""
FAISS vector store + semantic search over the SHL catalog.
"""

import json
import os
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class CatalogRetriever:
    """Loads a pre-built FAISS index and provides semantic search over SHL catalog items."""

    def __init__(self, index_dir: str = "faiss_index"):
        index_path = os.path.join(index_dir, "index.faiss")
        metadata_path = os.path.join(index_dir, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_dir}/. "
                "Run build_index.py first."
            )

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Build a set of all valid URLs for fast validation
        self.valid_urls = {item["link"] for item in self.metadata}

        print(f"CatalogRetriever loaded: {self.index.ntotal} items, "
              f"{len(self.valid_urls)} unique URLs")

    def semantic_search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Embed query with text-embedding-004 (RETRIEVAL_QUERY task),
        normalize, and return top_k metadata dicts from FAISS.
        """
        # Embed query
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        query_vector = np.array([result["embedding"]], dtype="float32")

        # Normalize for inner-product search
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["score"] = float(scores[0][i])
            results.append(item)

        return results

    def format_for_prompt(self, items: list[dict]) -> str:
        """
        Format a list of catalog items for injection into the LLM prompt.
        Returns newline-separated item summaries.
        """
        lines = []
        for item in items:
            desc = item.get("description", "")[:200]
            line = (
                f"Name: {item['name']} | "
                f"URL: {item['link']} | "
                f"Type: {item['test_type']} | "
                f"Duration: {item.get('duration', '')} | "
                f"Description: {desc}"
            )
            lines.append(line)
        return "\n".join(lines)

    def is_valid_url(self, url: str) -> bool:
        """Check if a URL exists in the catalog."""
        return url in self.valid_urls
