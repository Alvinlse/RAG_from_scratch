import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
from tqdm import tqdm

from semantic_search import retrieve_relevant_resources, embeddings, page_and_chunk, device

use_cuda = torch.cuda.is_available()
print(f"[INFO] CUDA available: {use_cuda}")

# 4-bit quantization is only supported on CUDA
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
) if use_cuda else None

# flash_attention_2 requires CUDA with compute capability >= 8.0
if use_cuda and is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")

model_id = "meta-llama/Llama-2-7b-chat-hf"
print(f"[INFO] Using model_id: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    attn_implementation=attn_implementation
)

if not use_cuda:
    llm_model.to(device)


def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])


def get_model_mem_size(model: torch.nn.Module):
    """Get how much memory a PyTorch model takes up."""
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    model_mem_bytes = mem_params + mem_buffers
    return {
        "model_mem_bytes": model_mem_bytes,
        "model_mem_mb": round(model_mem_bytes / (1024**2), 2),
        "model_mem_gb": round(model_mem_bytes / (1024**3), 2),
    }


def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """Format a query and retrieved context chunks into a prompt for the LLM."""
    context = "\n\n".join([f" {item['sentence_chunk']}" for item in context_items])
    prompt = f"""Based on the following context from a research paper, answer the user's question as clearly and thoroughly as possible.
If the context does not contain the answer, say so honestly.

Context:
{context}

Question: {query}
Answer:"""
    return prompt


def ask(query: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        n_resources_to_return: int = 5,
        return_answer_only: bool = True):
    """
    Full RAG pipeline: retrieve relevant chunks, augment prompt, generate answer.
    Returns the generated answer (and optionally the context items).
    """
    # 1. Retrieve relevant context
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    context_items = [page_and_chunk[i] for i in indices.tolist()]

    # 2. Build augmented prompt
    prompt = prompt_formatter(query=query, context_items=context_items)

    # 3. Tokenize and send to device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # 4. Generate
    with torch.inference_mode():
        with tqdm(total=max_new_tokens, desc="Generating answer", unit="token") as pbar:
            outputs = llm_model.generate(
                **input_ids,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                max_length=None,
            )
            pbar.update(outputs.shape[-1] - input_ids["input_ids"].shape[-1])

    # 5. Decode — strip the prompt from the output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output[len(tokenizer.decode(input_ids["input_ids"][0], skip_special_tokens=True)):].strip()

    if return_answer_only:
        return answer
    return answer, context_items


if __name__ == "__main__":
    query = "What are the main contributions of the paper?"
    print(f"Query: {query}\n")
    answer = ask(query)
    print(f"Answer:\n{answer}")
