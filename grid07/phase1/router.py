"""
phase1/router.py
----------------
Phase 1 – Vector-Based Persona Matching (The Router)

Flow:
  1. Three bot personas are embedded and stored in an in-memory ChromaDB collection.
  2. route_post_to_bots() embeds an incoming post, queries the vector store, and
     returns only the bots whose cosine similarity to the post exceeds `threshold`.

Dependencies:
  pip install chromadb sentence-transformers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Bot Personas
# ---------------------------------------------------------------------------
BOT_PERSONAS = {
    "bot_a": (
        "Tech Maximalist",
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space exploration. "
        "I dismiss regulatory concerns.",
    ),
    "bot_b": (
        "Doomer / Skeptic",
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature.",
    ),
    "bot_c": (
        "Finance Bro",
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI.",
    ),
}

# ---------------------------------------------------------------------------
# Embedding model (local, no API key needed)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small & fast; swap for a larger model if needed

embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# ---------------------------------------------------------------------------
# Build in-memory ChromaDB collection with persona embeddings
# ---------------------------------------------------------------------------
def build_persona_store() -> chromadb.Collection:
    """
    Creates an ephemeral ChromaDB collection and upserts each bot's persona text.
    Returns the populated collection.
    """
    client = chromadb.EphemeralClient()                        # purely in-memory
    collection = client.get_or_create_collection(
        name="bot_personas",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},                    # use cosine distance
    )

    documents, ids, metadatas = [], [], []
    for bot_id, (label, persona_text) in BOT_PERSONAS.items():
        documents.append(persona_text)
        ids.append(bot_id)
        metadatas.append({"label": label})

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    print(f"[Phase 1] Loaded {len(ids)} bot personas into the vector store.")
    return collection


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------
def route_post_to_bots(
    post_content: str,
    collection: chromadb.Collection,
    threshold: float = 0.35,          # <-- NOTE: ChromaDB cosine *distance* ∈ [0,2];
                                       #     distance < threshold → high similarity.
                                       #     0.35 ≈ cosine similarity > 0.82 for MiniLM.
) -> list[dict]:
    """
    Embeds `post_content` and queries the persona vector store.

    ChromaDB stores cosine *distance* (lower = more similar, range 0–2).
    We convert: similarity = 1 - (distance / 2)  so that the caller can
    reason in familiar [0, 1] similarity terms.

    Args:
        post_content: The incoming social-media post string.
        collection:   The ChromaDB collection with bot personas.
        threshold:    Minimum cosine *similarity* (0–1) required to route.
                      Default 0.35 works well for all-MiniLM-L6-v2; increase
                      to 0.85 if using a stronger embedding model.

    Returns:
        List of dicts: [{"bot_id": ..., "label": ..., "similarity": ...}, ...]
        Only bots that exceed the threshold are returned.
    """
    results = collection.query(
        query_texts=[post_content],
        n_results=len(BOT_PERSONAS),          # retrieve all bots
        include=["distances", "metadatas"],
    )

    matched_bots = []
    distances  = results["distances"][0]       # one query → first (only) list
    metadatas  = results["metadatas"][0]
    ids        = results["ids"][0]

    for bot_id, meta, dist in zip(ids, metadatas, distances):
        # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1]
        similarity = 1.0 - (dist / 2.0)
        if similarity >= threshold:
            matched_bots.append({
                "bot_id": bot_id,
                "label": meta["label"],
                "similarity": round(similarity, 4),
            })

    # Sort by similarity descending for readability
    matched_bots.sort(key=lambda x: x["similarity"], reverse=True)
    return matched_bots


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    collection = build_persona_store()

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits a new all-time high as ETF inflows surge this quarter.",
        "New study shows social media algorithms are designed to maximise outrage.",
        "NASA announces a crewed Mars mission funded by a private tech billionaire.",
        "Interest rates unchanged; Fed signals no cuts until Q3, yield curve flattens.",
    ]

    print("\n" + "=" * 70)
    print("PHASE 1 – ROUTING RESULTS")
    print("=" * 70)

    for post in test_posts:
        matched = route_post_to_bots(post, collection)
        print(f"\nPost : {post}")
        if matched:
            for m in matched:
                print(f"  ✅ {m['bot_id']} ({m['label']}) — similarity: {m['similarity']}")
        else:
            print("  ❌ No bot matched (similarity below threshold).")
