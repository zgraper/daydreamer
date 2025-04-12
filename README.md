# 🌌 Daydreamer AI

**Daydreamer AI** is an experimental creative writing and speculative thought generator powered by Large Language Models (LLMs). It takes a theme you choose and uses your own writings as a source of inspiration to explore imaginative, science-driven ideas in a 10-step sequence. Each step takes on a different style, from philosophical to interdimensional, creating a unique stream of ideas. At the end, it summarizes and highlights the most original or thought-provoking insights.

---

## 🧠 How It Works

1. **Vector Indexing with FAISS**  
   Your personal text or book (in `book_compilation.txt`) is split into overlapping chunks and embedded using `sentence-transformers`. These are indexed using FAISS for fast similarity search.

2. **Prompt Retrieval**  
   The `daydreamer.py` script lets you input a theme. Based on that theme, the program retrieves relevant excerpts from your book to inspire the dream.

3. **Multi-Step Dreaming**  
   The program generates 10 "dream steps," each using a different style:
   - Scientific Hypothesis
   - Philosophical Inquiry
   - Futuristic Scenario
   - Cosmic Engineering
   - and more...

4. **Summarization and Reflection**  
   A final LLM prompt creates a summary of the dream, highlighting original insights and reflecting on its themes.

5. **Saved Output**  
   Each dream is saved as a `.json` file, including:
   - Theme
   - Retrieved context
   - All 10 steps (with style)
   - Summary

---

## 🚀 Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/daydreamer-ai.git
cd daydreamer-ai

2. Install Requirements

pip install -r requirements.txt

    ✅ Make sure you also have FAISS and sentence-transformers installed.

3. Add Your Book

Place your compiled book into a text file named:

book_compilation.txt

4. Create the Index

Run this to generate your local vector database:

python index_creator.py

5. Generate a Daydream

python daydreamer.py

You'll be asked to enter a theme (e.g., time travel, quantum language, AI spirituality), and then the dreaming begins.
🔐 HuggingFace API

You’ll need an API key for HuggingFace’s Inference API. Set it in daydreamer.py:

HF_API_TOKEN = "your_token_here"

You can get one by signing up at: https://huggingface.co/settings/tokens
