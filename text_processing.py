"""
1. Document/ text pricessing and embedding creation
"""

"""
to do list:
1. Import PDF file.
2. process text for embedding (e.g. split into chunk of sentences. )
3. Embed text chunk with embedding model.
4. save embeddings for later use.
"""

from tqdm.auto import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from parser import chunk_info

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
 Semantic chunking — method recommended by Kshirsagar (2024):
   "Enhancing RAG Performance Through Chunking and Text Splitting Techniques"

 Algorithm (per the paper):
   1. For each sentence i, build a *sentence group*: the window of sentences
      surrounding i (buffer_size before + sentence i + buffer_size after).
      Joining neighbours gives each embedding more local context than a single
      sentence would provide.
   2. Embed every group in one batched call.
   3. Compute the cosine distance between embeddings of consecutive groups.
      Low distance → same topic; high distance → topic shift.
   4. Split at positions where the distance exceeds the breakpoint_percentile
      threshold — this adapts to each page's own distribution of distances.
"""
_chunk_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device=device)

def semantic_chunk(sentences: list[str],
                   buffer_size: int = 1,
                   breakpoint_percentile: int = 95,
                   min_chunk_sentences: int = 3,
                   max_chunk_sentences: int = 20) -> list[list[str]]:
    """Split sentences into semantically coherent chunks using sentence groups.

    Args:
        sentences: list of sentence strings for one page.
        buffer_size: number of neighbouring sentences to include on each side
            when building a sentence group for embedding (paper recommends 1).
        breakpoint_percentile: cosine-distance percentile above which a
            boundary is inserted (higher → fewer, larger chunks).
        min_chunk_sentences: never split a chunk shorter than this.
        max_chunk_sentences: hard cap — always split after this many sentences.
    """
    import numpy as np

    if len(sentences) <= min_chunk_sentences:
        return [sentences]

    # Step 1 — build sentence groups centred on each sentence.
    groups = []
    for i in range(len(sentences)):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        groups.append(' '.join(sentences[start:end]))

    # Step 2 — embed all groups in one batched call.
    group_embeddings = _chunk_model.encode(groups, convert_to_numpy=True)

    # Step 3 — cosine distances between consecutive group embeddings.
    norms = np.linalg.norm(group_embeddings, axis=1, keepdims=True)
    normed = group_embeddings / np.where(norms == 0, 1, norms)
    cosine_sims = (normed[:-1] * normed[1:]).sum(axis=1)   # shape: (n-1,)
    cosine_dists = 1 - cosine_sims

    # Step 4 — threshold at the given percentile of this page's distances.
    threshold = float(np.percentile(cosine_dists, breakpoint_percentile))

    chunks: list[list[str]] = []
    current: list[str] = []

    for i, sentence in enumerate(sentences):
        current.append(sentence)
        if i < len(sentences) - 1:
            gap_is_large = cosine_dists[i] >= threshold
            chunk_too_long = len(current) >= max_chunk_sentences
            chunk_long_enough = len(current) >= min_chunk_sentences
            if (gap_is_large and chunk_long_enough) or chunk_too_long:
                chunks.append(current)
                current = []

    if current:
        # Merge a tiny trailing fragment into the previous chunk.
        if chunks and len(current) < min_chunk_sentences:
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    return chunks


for item in tqdm(chunk_info, desc="Semantic chunking"):
    item['sentences_chunk'] = semantic_chunk(item['text'].split('. '))

# splitting each chunk into its own item.
page_and_chunk = []
for item in chunk_info:
    for sentence_chunk in item['sentences_chunk']:
        joined_sentences_chunk = ' '.join(sentence_chunk)
        page_and_chunk.append({
            'sentence_chunk': joined_sentences_chunk,
            'chunk_char_count': len(joined_sentences_chunk),
            'chunk_word_count': len(joined_sentences_chunk.split()),
            'chunk_token_count': len(joined_sentences_chunk) / 4,  # ~4 chars per token
        })


"""
embedding text chunk  (reuses the model already loaded for semantic chunking)
"""
all_chunks = [item['sentence_chunk'] for item in page_and_chunk]
all_embeddings = _chunk_model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
for item, emb in zip(page_and_chunk, all_embeddings):
    item['embedding'] = emb

"""
 for small dataset, we can store the embedding vector in csv file.
 for larger dataset,  we need a vectorbase for storage such as Qdrant, Chromadb , etc.
"""
text_chunk_and_embedding_df = pd.DataFrame(page_and_chunk)
text_chunk_and_embedding_df_save_path = 'text_chunk_and_embedding_df.csv'
text_chunk_and_embedding_df.to_csv(text_chunk_and_embedding_df_save_path, index=False)
