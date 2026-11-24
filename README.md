# 🤖 RAG APPLICATION

A modern, local **Retrieval-Augmented Generation (RAG)** application powered by **LangChain**, **Ollama**, **ChromaDB**, and a sleek **ChatGPT-inspired UI** built with **Streamlit**.

---

## 🌟 Key Features

- **ChatGPT Dark-Mode Aesthetic**: Styled to look and feel like ChatGPT with dark charcoal tones, floating capsule input, and smooth user/assistant chat bubbles.
- **Multi-Source Knowledge Base**:
  - 📄 **PDF Documents**: Automatic extraction via PyPDFLoader.
  - 📝 **Text Files**: Local TXT file ingestion.
  - 🌐 **Web URLs**: Live webpage crawling and indexing using WebBaseLoader.
- **Fast Vector Search**: Chunking with `RecursiveCharacterTextSplitter` and persistent vector storage using **ChromaDB**.
- **100% Local & Private AI**: Runs entirely offline on your machine using **Ollama**:
  - **LLM**: `llama3.1`
  - **Embeddings**: `nomic-embed-text`
- **Instant Citations & Sources**: View the exact chunks, document names, and page numbers referenced by the model.
- **Typewriter Effect**: Streaming responses word-by-word just like ChatGPT.
- **One-Click New Chat & History**: Clear conversation or start fresh anytime.

---

## 📁 Project Structure

```
RAG APPLICATION/
├── app.py              # Main ChatGPT-style Streamlit UI & interaction
├── rag.py              # Core RAG pipeline (loaders, splitters, ChromaDB, Ollama)
├── requirements.txt    # Python dependencies
├── run_streamlit.bat   # Windows 1-click startup launcher
└── uploads/            # Temporary directory for uploaded documents
```

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.10+** and **[Ollama](https://ollama.com/)** installed.

Pull the required Ollama models:
```bash
ollama run llama3.1
ollama pull nomic-embed-text
```

### 2. Clone the Repository
```bash
git clone https://github.com/priyanshu-077980/RAG-APPLICATION.git
cd RAG-APPLICATION
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application

#### Option A: 1-Click Launcher (Windows)
Double-click `run_streamlit.bat`.

#### Option B: Terminal Command
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🛠️ How It Works

1. **Document Loading & Chunking**: When documents (PDF, TXT, or URL) are added, text is split into chunks of 500 characters with 50-character overlap.
2. **Embedding & Storage**: Vectors are generated using `nomic-embed-text` and stored in ChromaDB.
3. **Retrieval**: When you ask a question, ChromaDB retrieves the top 3 most relevant context chunks.
4. **Generation**: The context and prompt are fed into `llama3.1` to produce an accurate, context-grounded response.

---

## 📜 License
MIT License. Free to use, modify, and distribute.
