import random
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from datetime import datetime

# === CONFIG ===
INDEX_PATH = "faiss_index.index"
CHUNKS_PATH = "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_API_TOKEN = "your_token_here"  # Insert information here

DAYDREAM_STYLES = [
    "Philosophical Inquiry",
    "Scientific Hypothesis",
    "Futuristic Scenario",
    "Technological Analogy",
    "Metaphysical Reflection",
    "Neuroscience Speculation",
    "Interdimensional Thought Experiment",
    "Evolutionary Lens",
    "Cosmic Engineering",
    "Philosci-Fi Fusion",
]

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# === Load index and chunks ===
def load_index_and_chunks():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# === Retrieve context from FAISS ===
def retrieve_context(query, index, chunks, embed_model, top_k=5):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

# === Generate one step ===
def generate_step(prompt, trim_prompt=True):
    response = requests.post(
        HF_MODEL_ENDPOINT,
        headers=HEADERS,
        json={"inputs": prompt}
    )

    try:
        output = response.json()[0]["generated_text"]
    except Exception as e:
        return f"[ERROR] {e} - Raw output: {response.text}"

    # Remove the prompt from the output if the model echoes it
    if trim_prompt and prompt in output:
        output = output.replace(prompt, "").strip()

    return output.strip()

# === Generate one full dream ===
def generate_daydream(theme, index, chunks, embed_model):
    retrieved_context = retrieve_context(theme, index, chunks, embed_model)
    base_context = "\n\n".join(retrieved_context)
    
    steps = []
    current_context = theme
    for i in range(10):
        style = random.choice(DAYDREAM_STYLES)
        prompt = f"""
You are an AI engaging in speculative, grounded daydreaming.
Your current style is: {style}
Theme: {theme}

Here is some inspirational context from books:
{base_context}

Continue the dream in a way that matches the style. Expand on the previous thoughts or add a new dimension:
Current dream: {current_context}
        """
        step_output = generate_step(prompt)
        steps.append({
            "style": style,
            "step_text": step_output.strip()
        })
        current_context += "\n" + step_output.strip()
    
    return retrieved_context, steps

# === Summarize the dream ===
def summarize_dream(theme, steps):
    combined = "\n\n".join([f"{i+1}. [{s['style']}] {s['step_text']}" for i, s in enumerate(steps)])
    summary_prompt = f"""
You are a philosophical AI summarizing a speculative 10-part dream. Identify the most original, insightful, or thought-provoking ideas from the following stream of thoughts. Also reflect briefly on the themes that emerged.

Theme: {theme}
Daydream:
{combined}
    """
    return generate_step(summary_prompt)

# === Save output ===
def save_dream(theme, context, steps, summary):
    output = {
        "theme": theme,
        "retrieved_context": context,
        "steps": steps,
        "summary": summary
    }
    filename = f"daydream_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n🌌 Dream saved as: {filename}")

# === Main Program ===
def main():
    index, chunks = load_index_and_chunks()
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("💭 Welcome to the Daydreamer.")
    theme = input("Enter a theme for this dream (e.g., time, black holes, language): ")

    print("\n🔍 Retrieving inspiration from your book...")
    context, steps = generate_daydream(theme, index, chunks, model)

    print("\n🧠 Summarizing and reflecting...")
    summary = summarize_dream(theme, steps)

    print("\n✨ Final Summary:\n")
    print(summary)

    save_dream(theme, context, steps, summary)

if __name__ == "__main__":
    main()
