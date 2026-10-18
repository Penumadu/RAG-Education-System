# PDF RAG Knowledge Assistant

A production-grade, strictly grounded Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Streamlit**.

---

## 🌟 Key Features

1. **Enterprise Document Library & Lifecycle Management**:
   - **Multi-document ingestion**: Upload one or multiple PDFs incrementally at any time without overwriting existing data.
   - **Granular document deletion**: View all indexed PDFs in the library and delete individual documents without wiping the whole database.
   - **Persistent across restarts**: All vectors and document catalogs are stored in persistent ChromaDB storage (`.chroma_data`), preserved across server restarts and next-day sessions.
   - **Duplicate prevention & batch progress**: Automatic detection of existing files and real-time chunking progress bars.

2. **Professional UI & Modern Aesthetics**:
   - Custom `.streamlit/config.toml` design with dark/light mode compatibility, modern typography (Inter, JetBrains Mono), 8px border radii, and KPI metrics cards.
   - Sidebar tab navigation separating upload, library management, and runtime settings.
   - Clickable starter query pills for instant exploration.
   - Expandable source citation cards with document names, page numbers, and relevance confidence percentages.

3. **Strict Grounding & Anti-Hallucination Guardrails**:
   - The agent is constrained by system prompts and similarity score thresholding.
   - If the answer is not present in the uploaded PDFs, it explicitly responds:
     > *"I don't have information about that in the uploaded documents."*

4. **LangChain Agent Harness & Memory**:
   - Built on `create_agent` with tool binding and `@wrap_tool_call` error-handling middleware.
   - Multi-turn conversational memory powered by LangGraph's `MemorySaver` checkpointer keyed by `thread_id`.

5. **100% Local & Private**:
   - LLM: `qwen3.5:9b` via Ollama
   - Embeddings: `nomic-embed-text` via Ollama
   - Vector Store: `Chroma` local persistent database

---

## 🚀 Quickstart

### 1. Prerequisites
Ensure [Ollama](https://ollama.ai) is running locally:
```bash
ollama serve
```

Ensure the models are pulled:
```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### 2. Launch the Application
Run the Streamlit application:
```bash
uv run streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### 3. Run Automated Tests
```bash
uv run python test_rag.py
```

---

## 📂 Project Structure

```
.
├── app.py                # Streamlit Web UI & chat interface
├── rag_engine.py         # PDF parsing, chunking, embedding & Chroma retrieval
├── agent.py              # LangChain agent harness, memory checkpointer & middleware
├── test_rag.py           # Self-contained end-to-end test suite
├── .streamlit/
│   └── config.toml       # Streamlit UI theme configuration
├── pyproject.toml        # Project dependencies
└── README.md             # Documentation
```
