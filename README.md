# The Forbidden LibrAIry Agentic RAG Chatbot

An intelligent conversational AI chatbot that answers questions about the Harry Potter book universe using **Agentic RAG** (Retrieval-Augmented Generation). The chatbot autonomously decides when to retrieve information from the Harry Potter books and provides accurate, contextual answers.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red.svg)

## What Makes This Special?

This isn't just a simple question-answering system. It's an **agentic system** where an AI agent makes intelligent decisions about:
- **When** to retrieve information from the knowledge base
- **What** queries to use for retrieval
- **How** to synthesize retrieved information with its reasoning
- **Whether** multiple retrievals are needed for complex queries

## Cool Features

### 1. **Agentic RAG Architecture**
Unlike traditional RAG systems that retrieve for every query, this chatbot uses an **AI agent** that intelligently decides when retrieval is necessary. This means:
- For simple greetings or general queries, it responds directly without unnecessary database calls
- For Harry Potter-specific questions, it automatically retrieves relevant context
- It can perform multiple retrievals for complex, multi-part questions

### 2. **Semantic Search with Vector Embeddings**
- Uses OpenAI's `text-embedding-3-small` model to create high-quality vector embeddings
- Stores embeddings in **Supabase Vector Store** for fast, scalable similarity search
- Retrieves the most contextually relevant passages, not just keyword matches

### 3. **Conversational Memory**
- Maintains full chat history throughout the conversation
- Can reference previous messages and maintain context
- Enables follow-up questions without repeating information

### 4. **Interactive Streamlit UI**
- Beautiful, user-friendly chat interface
- Real-time streaming responses
- Chat history visualization
- Easy to use for both developers and end-users

### 5. **Modular Architecture**
- Separate ingestion pipeline for document processing
- Reusable components for embedding and retrieval
- Easy to extend with additional tools or data sources

## 🏗️ How It Works

### System Architecture

```
┌─────────────────┐
│  Harry Potter   │
│   PDF Books     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Document Ingestion     │
│  - PDF Loading          │
│  - Text Chunking        │
│  - Vector Embeddings    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Supabase Vector Store  │
│  (Persistent Storage)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   LangChain Agent       │
│   - Tool Calling        │
│   - Decision Making     │
│   - GPT-4o LLM          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Streamlit Chat UI     │
│   (User Interface)      │
└─────────────────────────┘
```

### Technical Pipeline

1. **Document Ingestion (`ingest_in_db.py`)**
   - Loads Harry Potter PDF from the `documents/` directory
   - Splits text into 1000-character chunks with 100-character overlap
   - Generates vector embeddings using OpenAI's embedding model
   - Stores chunks and embeddings in Supabase vector database

2. **Retrieval Tool**
   ```python
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve structured passages from the Harry Potter books."""
    retrieved_docs = vector_store.similarity_search(query, k=4)

    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata or {}
        page = meta.get("page_label") or meta.get("page") or "unknown page"
        parts.append(
            f"--- Passage {i} (page {page}) ---\n{doc.page_content.strip()}"
        )

    context_text = "\n\n".join(parts)
    return context_text, retrieved_docs

   ```
   - Performs semantic similarity search
   - Includes source metadata for transparency

3. **Agent Decision-Making**
   - Uses OpenAI's function calling capabilities
   - Agent decides autonomously whether to invoke the retrieval tool
   - Synthesizes retrieved information into coherent answers
   - Powered by GPT-4o for high-quality responses

4. **User Interface**
   - **Command-Line**: `agentic_rag.py` for quick testing
   - **Web UI**: `streamlit_agentic_rag.py` for interactive conversations

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Supabase account (with vector store enabled)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Harry-Potter-Chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```

4. **Set up Supabase Vector Store**
   
   In your Supabase SQL editor, run:
   ```sql
   -- Enable the pgvector extension
         --create extension if not exists vector;
         
         -- Drop old function and table if they exist
         drop function if exists match_documents(vector, float, int);
         drop table if exists documents;
         
         -- Create the documents table with UUID id
         create table documents (
           id uuid primary key,
           content text,
           metadata jsonb,
           embedding vector(1536)
         );
         
         -- Create a function for similarity search
         create or replace function match_documents(
           query_embedding vector(1536),
           match_threshold float,
           match_count int
         )
         returns table (
           id uuid,
           content text,
           metadata jsonb,
           similarity float
         )
         language sql stable
         as $$
           select
             documents.id,
             documents.content,
             documents.metadata,
             1 - (documents.embedding <=> query_embedding) as similarity
           from documents
           where 1 - (documents.embedding <=> query_embedding) > match_threshold
           order by similarity desc
           limit match_count;
         $$;
         --

5. **Ingest the Harry Potter documents**
   ```bash
   python ingest_in_db.py
   ```
   This will process the PDF and populate your vector store.

### Running the Chatbot

**Option 1: Streamlit Web UI (Recommended)**
```bash
streamlit run streamlit_agentic_rag.py
```
Then open your browser to `http://localhost:8501`

**Option 2: Command Line**
```bash
python agentic_rag.py
```

## 💬 Sample Questions to Ask

- "Name everyone in Ron's family"
- "What is the prophecy about Harry and Voldemort?"
- "How did Harry defeat Voldemort?"
- "What are the four Hogwarts houses and their traits?"
- "Explain how Quidditch is played"
- "What are the Deathly Hallows?"
- "What is a Horcrux and how does it work?"
- "What is the Marauder's Map"
- "What is the relationship between Harry, Snape, and his parents?"
- "How did each of Voldemort's Horcruxes get destroyed?"

## 🛠️ Technical Stack

- **LangChain**: Orchestration framework for LLM applications
- **OpenAI GPT-4o**: Large language model for reasoning and generation
- **OpenAI Embeddings**: Text embedding model (text-embedding-3-small)
- **Supabase**: PostgreSQL database with pgvector extension
- **Streamlit**: Web UI framework
- **Python**: Core programming language

## 📁 Project Structure

```
Harry-Potter-Chatbot/
├── documents/
│   └── harrypotter.pdf           # Source material
├── agentic_rag.py                # Command-line chatbot
├── streamlit_agentic_rag.py      # Streamlit web UI
├── ingest_in_db.py               # Document ingestion pipeline
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
└── README.md                     # This file
```

## 📝 License

This project is for educational purposes. Harry Potter content is owned by J.K. Rowling and Warner Bros.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

---

