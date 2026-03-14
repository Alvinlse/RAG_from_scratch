import os
import torch
import numpy as np
import pandas as pd
import textwrap
from sentence_transformers import SentenceTransformer, util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load embeddings from CSV
_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_chunk_and_embedding_df.csv')
text_chunk_and_embedding_df = pd.read_csv(_csv_path).dropna(subset=['sentence_chunk'])
text_chunk_and_embedding_df['embedding'] = text_chunk_and_embedding_df['embedding'].apply(
    # BUG FIX: np.fromstring with sep=" " doesn't handle newlines in multi-line
    # numpy string representations (768-dim arrays wrap across lines).
    # str.split() splits on all whitespace including newlines.
    lambda x: np.array(x.strip("[]").split(), dtype=np.float32)
)

embeddings = torch.tensor(np.stack(text_chunk_and_embedding_df['embedding'].to_numpy()), dtype=torch.float32).to(device)
page_and_chunk = text_chunk_and_embedding_df.to_dict(orient='records')

# Load the embedding model once at module level
embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device=device)


def print_wrapped(text: str, wrap_length: int = 80):
    """Print text wrapped at wrap_length characters."""
    wrapped = textwrap.fill(text, wrap_length)
    print(wrapped)


def retrieve_relevant_resources(query: str,
                                embeddings: torch.Tensor,
                                model: SentenceTransformer = None,
                                n_resources_to_return: int = 5):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """
    if model is None:
        model = embedding_model

    # Embed the query — force float32 to match stored embeddings
    query_embedding = model.encode(query, convert_to_tensor=True).to(device).to(torch.float32)

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices


def print_top_results_and_scores(query: str,
                                 embeddings: torch.Tensor,
                                 page_and_chunk: list[dict] = page_and_chunk,
                                 n_resources_to_return: int = 5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.
    """
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    print(f"Query: {query}\n")
    print("Results:")
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        idx = index.item()  # BUG FIX: convert 0-dim tensor to plain Python int
        print_wrapped(page_and_chunk[idx]["sentence_chunk"])
        print("\n")


if __name__ == "__main__":
    query = 'Experiment'
    print_top_results_and_scores(query=query, embeddings=embeddings)
