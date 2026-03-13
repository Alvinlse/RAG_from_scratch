# RAG From Scratch

A personal project to learn and understand **Retrieval-Augmented Generation (RAG)** by building it from scratch. The system answers questions about uploaded PDF documents by combining semantic search with a local LLM.

---

## What Is This?

This project implements a full RAG pipeline without relying on high-level frameworks like LangChain or LlamaIndex. The goal is to understand every step — from PDF parsing and embedding generation to semantic retrieval and LLM-based answer generation.

The current demo document is a research paper: *Learning to Optimize Tensor Programs* (stored in `docs/`).

---

## Architecture

```
PDF → Extract Text → Split Sentences → Chunk (10 sentences) → Embed → CSV Cache
                                                                          ↓
User Query → Embed Query → Semantic Search → Top 5 Chunks → Format Prompt → LLM → Answer
```

### Modules

| File | Role |
|---|---|
| [text_processing.py](text_processing.py) | Parse PDF, chunk text, generate and save embeddings |
| [semantic_search.py](semantic_search.py) | Load embeddings, embed queries, retrieve top-k relevant chunks |
| [generate.py](generate.py) | Load Llama-2, format prompt with context, generate answers |
| [main.py](main.py) | Interactive Q&A loop (entry point) |

---

## Requirements

- Python 3.10
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

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Hugging Face access (for Llama-2)

Llama-2 requires accepting Meta's license on Hugging Face and logging in:

```bash
huggingface-cli login
```

Then accept the model license at: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

---

## Running the Project

### Step 1 — Process the document (run once)

This extracts text from the PDF, splits it into sentence chunks, generates embeddings, and saves everything to a CSV file.

```bash
python text_processing.py
```

Output: `text_chunk_and_embedding_df.csv`

> Skip this step if the CSV file already exists in the repo — it is pre-generated.

### Step 2 — Start the interactive Q&A

```bash
python main.py
```

You will see a prompt to enter your question. The system will retrieve the most relevant chunks from the document and generate an answer using Llama-2.

```
Ask a question (or type 'quit' to exit): What is TVM?
```

---

## Using Your Own Document

1. Place your PDF in the `docs/` folder.
2. Update the `pdf_path` variable in [text_processing.py](text_processing.py) to point to your file.
3. Run `python text_processing.py` to regenerate the embeddings CSV.
4. Run `python main.py` to start asking questions.

---

## Models Used

| Purpose | Model |
|---|---|
| Embeddings | `all-mpnet-base-v2` (sentence-transformers) |
| Answer generation | `meta-llama/Llama-2-7b-chat-hf` (4-bit quantized) |

---

## Project Structure

```
RAG_from_scratch/
├── docs/
│   └── Learning to Optimize Tensor Programs.pdf
├── main.py                          # Entry point — interactive Q&A
├── text_processing.py               # PDF → chunks → embeddings → CSV
├── semantic_search.py               # Query embedding + retrieval
├── generate.py                      # LLM prompt formatting + generation
├── text_chunk_and_embedding_df.csv  # Pre-computed embeddings cache
├── pyproject.toml
└── requirement.txt
```
