# RAG From Scratch

A personal project to learn and understand **Retrieval-Augmented Generation (RAG)** by building it from scratch. The system answers questions about uploaded PDF documents by combining semantic search with a local LLM.

---

## What Is This?

This project implements a full RAG pipeline without relying on high-level frameworks like LangChain or LlamaIndex. The goal is to understand every step — from PDF parsing and embedding generation to semantic retrieval and LLM-based answer generation.

The current demo document is a research paper: *Learning to Optimize Tensor Programs* (stored in `docs/`).

---

## Architecture

```
PDF → PyMuPDF Parser (section detection) → Semantic Chunking → Embed → CSV Cache
                                                                              ↓
User Query → Embed Query → Semantic Search (dot-product) → Top 5 Chunks → Format Prompt → LLM → Answer
```

### Modules

| File | Role |
|---|---|
| [parser.py](parser.py) | Parse PDF into section-level chunks using PyMuPDF (font-size heading detection, table/figure handling) |
| [text_processing.py](text_processing.py) | Apply semantic chunking to parsed sections, generate embeddings, save to CSV |
| [semantic_search.py](semantic_search.py) | Load embeddings, embed queries, retrieve top-k relevant chunks via dot-product similarity |
| [generate.py](generate.py) | Load Llama-2, format augmented prompt with retrieved context, generate answers |
| [main.py](main.py) | Interactive Q&A loop (entry point) |

---

## Chunking Strategy

Text is split using **semantic chunking** as described in Kshirsagar (2024), *"Enhancing RAG Performance Through Chunking and Text Splitting Techniques"*:

1. For each sentence, build a **sentence group** — a window of neighbouring sentences (buffer of ±1) to give each embedding more local context.
2. Embed all groups in one batched call using `all-mpnet-base-v2`.
3. Compute cosine distance between consecutive group embeddings. High distance = topic shift.
4. Split at positions where the distance exceeds the **95th percentile** of distances on that page — adapts to each page's own topic distribution.

Chunk size is constrained to a minimum of 3 sentences and a hard cap of 20 sentences.

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended (Llama-2 7B with 4-bit quantization)
- [UV](https://github.com/astral-sh/uv) package manager

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RAG_from_scratch
```

### 2. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate       # Linux / macOS
# or
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
uv sync
```

Key dependencies include: `sentence-transformers`, `transformers`, `torch`, `pymupdf`, `pandas`, `tqdm`, `bitsandbytes`.

### 4. Hugging Face access (for Llama-2)

Llama-2 requires accepting Meta's license on Hugging Face and logging in:

```bash
huggingface-cli login
```

Then accept the model license at: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

---

## Running the Project

### Step 1 — Process the document (run once)

This parses the PDF into sections, applies semantic chunking, generates embeddings, and saves everything to a CSV file.

```bash
python text_processing.py
```

Output: `text_chunk_and_embedding_df.csv`

> Skip this step if the CSV file already exists in the repo — it is pre-generated.

### Step 2 — Start the interactive Q&A

```bash
python main.py
```

You will see a prompt to enter your question. The system retrieves the most relevant chunks and shows them before generating an answer with Llama-2.

```
Ask a question about the paper: What is TVM?

--- Retrieved context ---
Query: What is TVM?

Results:
Score: 0.8321
...

--- Generating answer ---

Answer:
...
```

---

## Using Your Own Document

1. Place your PDF in the `docs/` folder.
2. Update the `pdf_path` variable in [parser.py](parser.py) to point to your file.
3. Run `python text_processing.py` to regenerate the embeddings CSV.
4. Run `python main.py` to start asking questions.

---

## Models Used

| Purpose | Model |
|---|---|
| Embeddings | `all-mpnet-base-v2` (sentence-transformers) |
| Answer generation | `meta-llama/Llama-2-7b-chat-hf` (4-bit quantized via bitsandbytes) |

---

## Project Structure

```
RAG_from_scratch/
├── docs/
│   └── Learning to Optimize Tensor Programs.pdf
├── main.py                          # Entry point — interactive Q&A
├── parser.py                        # PDF → section chunks (PyMuPDF)
├── text_processing.py               # Section chunks → semantic chunks → embeddings → CSV
├── semantic_search.py               # Query embedding + dot-product retrieval
├── generate.py                      # LLM prompt formatting + generation
├── text_chunk_and_embedding_df.csv  # Pre-computed embeddings cache
├── pyproject.toml
└── requirement.txt
```
