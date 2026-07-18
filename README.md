# ATHERA
### A Research Paper Intelligence System

An end-to-end NLP pipeline that lets you **search, summarize, extract keywords from, and compare** research papers using natural language queries — powered by semantic search, transformer summarization, and an LLM agent that automatically selects the right tool for each query.

---

##  Overview

Finding relevant research papers—and then understanding them—doesn't scale when you're dealing with thousands of publications. This project builds a complete retrieval-augmented NLP pipeline over a corpus of **15,000 machine learning papers from arXiv**, then wraps it inside a conversational LLM agent so users can interact with the entire collection using plain English.

---

##  Pipeline

### 1. Data Loading & Cleaning
- **Dataset:** `CShorten/ML-ArXiv-Papers`
- Loaded paper titles and abstracts
- Removed duplicates and cleaned text

### 2. Semantic Embeddings
- **Model:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- Converts every paper into a **384-dimensional embedding**

### 3. Vector Search
- Built a **FAISS** inner-product index
- L2-normalized embeddings enable fast cosine similarity search
- Retrieves the most semantically relevant papers for any query

### 4. Summarization
- **Model:** `facebook/bart-large-cnn`
- Generates concise summaries of retrieved abstracts
- Summary length automatically adapts to input size

### 5. Keyword Extraction
- Powered by **KeyBERT**
- Extracts representative keywords and keyphrases from research papers

### 6. Named Entity Recognition
- Uses a general-purpose NER pipeline to identify:
  - Models
  - Datasets
  - Organizations and Institutions

### 7. LLM Agent Layer
- Built using **LangChain + Groq (Llama 3.1 8B)**
- Exposes retrieval, summarization, and keyword extraction as callable tools
- The agent automatically determines which tool(s) to invoke based on the user's query

---

##  Tech Stack

- Python
- Sentence-Transformers
- FAISS
- Hugging Face Transformers
- KeyBERT
- LangChain
- LangChain-Groq
- Pandas
- NumPy

---

##  Example Queries

```text
Find the top 3 research papers on Vision Transformers and summarize them.
```

```text
Extract the top 5 keywords from:
Deep Learning for Medical Image Reconstruction.
```

```text
Compare a paper about Vision Transformers with a paper about Convolutional Neural Networks.
```

---

##  Installation

```bash
pip install datasets sentence-transformers faiss-cpu keybert \
transformers==4.46.3 huggingface_hub==0.26.2 tokenizers==0.20.3 sentence-transformers==3.3.1 \
langchain langchain-community langchain-core langchain-huggingface langchain-groq
```

### Set your Groq API Key

**Linux / macOS**

```bash
export GROQ_API_KEY=your_key_here
```

**Windows (Command Prompt)**

```cmd
set GROQ_API_KEY=your_key_here
```

**Windows (PowerShell)**

```powershell
$env:GROQ_API_KEY="your_key_here"
```

Run the notebook from top to bottom. The embeddings and FAISS index are automatically cached after the first execution.

---

##  Author

**Atiksh Sharma**