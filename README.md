# CinephileGPT 🎬

**CinephileGPT** is a production-grade, local-first conversational AI assistant whose entire personality revolves around movies, cinema, directors, actors, and film history. The assistant behaves like a passionate, opinionated, and highly knowledgeable film buff—answering movie queries with witty cinematic analogies, while humorously deflecting non-movie-related requests (like coding or cooking queries) back into film discussions.

The system is designed with a modular architecture featuring a **FastAPI backend**, an **interactive single-page frontend**, a **hybrid RAG retrieval layer**, and an **asynchronous user memory consolidation system**.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Client]) <--> Frontend[Single Page Dashboard]
    Frontend <--> API[FastAPI Backend]
    
    %% Ingress & Intent
    API --> Intent[Intent Classifier]
    Intent --> |Non-Movie Query| Refusal[Humorous Refusal Generator]
    Intent --> |Movie Query| MemoryFetch[User Memory Extractor]
    MemoryFetch --> |Fetch Profile & Chat History| Postgres[(Local PostgreSQL)]
    
    %% Hybrid RAG
    MemoryFetch --> RAG[RAG Retrieval Engine]
    RAG --> |SQL Query Filters| Postgres
    RAG --> |Vector Search via all-MiniLM-L6-v2| Qdrant[(Local Qdrant File DB)]
    
    %% Response & Update
    RAG --> Composer[Prompt Composer]
    Composer --> LLM[Gemini 2.5 Flash]
    LLM --> Stream[Stream Engine]
    Stream --> API
    LLM --> |Extract Preferences (Real-time)| MemoryUpdater[Memory Consolidation]
    MemoryUpdater --> Postgres
```

---

## 🌟 Core Features

*   **Witty Cinephile Persona**: Converses like a dedicated film buff, referencing cinematography styles, director cuts, and actors, while humorously rejecting non-movie queries.
*   **Zero-Dependency Local Vector Search**: Uses local `sentence-transformers/all-MiniLM-L6-v2` embeddings combined with a file-based Qdrant client (`qdrant-client` path storage), removing any Docker container requirements.
*   **Hybrid RAG Pipeline**: Blends dense vector search matching with PostgreSQL relational table queries (strictly filtering out disliked genres dynamically, while keeping favorite genres as soft personalization parameters at the LLM level to allow unrestricted searching).
*   **Multi-Tiered Memory & Profiling**:
    *   *Short-term Memory*: Preserves the last 10 messages (5 full turns of alternating user/assistant dialogs) to maintain perfect conversational continuity.
    *   *Long-term Memory*: A real-time preference extraction mechanism that runs inline at the end of the streaming turn, utilizing Gemini API to extract newly mentioned preferences and update the PostgreSQL user profiles, immediately pushing a synchronization event (`memory_update`) to the frontend.
*   **Interactive Cinema Dashboard**:
    *   Streamed responses (SSE tokens) rendered inside clean dialogue blocks.
    *   Interactive movie cards rendered dynamically on search hits (with poster previews and direct "Add to Watchlist" quick-actions).
    *   Sidebars to list, load, and delete conversation threads, view watchlists, and manually edit preference profiles.

---

## 📂 Codebase Directory Layout

```
CinephileGPT/
├── backend/                      # FastAPI Backend
│   ├── app/
│   │   ├── api/                  # API routers (endpoints)
│   │   │   ├── auth.py           # JWT logins, signups, bcrypt hashes
│   │   │   ├── chat.py           # Chat session listings & SSE streaming
│   │   │   ├── memory.py         # Read/Write personalization profiles
│   │   │   └── movies.py         # Catalog lookup and watchlist edits
│   │   ├── core/                 # Configurations & DB bindings
│   │   │   ├── config.py         # Environment configurations parser
│   │   │   ├── database.py       # SQL engine & Qdrant lazy-loader
│   │   │   └── security.py       # Password checks & JWT token generation
│   │   ├── models/               # SQLAlchemy SQL schemas
│   │   │   ├── user.py           # Users table
│   │   │   ├── movie.py          # Movies & Watchlist tables
│   │   │   └── memory.py         # UserProfiles & Chat histories
│   │   ├── schemas/              # Pydantic validation schemas
│   │   │   ├── chat.py
│   │   │   ├── memory.py
│   │   │   └── movie.py
│   │   └── services/             # Core RAG, ML, & LLM orchestrations
│   │       ├── embeddings.py     # Local sentence-transformers encoder
│   │       ├── intent.py         # Zero-shot intent classification
│   │       ├── retrieval.py      # Hybrid RAG & Qdrant query filters
│   │       └── llm.py            # Streaming Gemini chats & memory updates
│   ├── scripts/                  # Utilities & validations
│   │   ├── seed_movies.py        # Relational and vector seeding script
│   │   └── test_backend.py       # Diagnostic validation script
│   ├── static/                   # Frontend SPA Dashboard
│   │   └── index.html            # Main single-file layout (HTML/CSS/JS)
│   ├── requirements.txt          # Backend dependencies list
│   └── .env                      # Active configurations (Ignored from Git)
└── .gitignore                    # Git file exclusions checklist
```

---

## 🚀 Installation & Launch Guide

### 1. Setup Your Virtual Environment
Navigate to the `backend/` directory, create a virtual environment, and activate it:

```bash
cd backend
python -m venv venv
```

*   **PowerShell**:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
*   **Command Prompt (cmd)**:
    ```cmd
    venv\Scripts\activate.bat
    ```

### 1.1. Configure Your IDE Interpreter (VS Code / Cursor)
To prevent your editor from complaining about missing imports (like `qdrant_client`, `sentence_transformers`, etc.), configure your workspace to use the virtual environment's Python interpreter.

Create or check the `.vscode/settings.json` file in the workspace root:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/backend/venv/Scripts/python.exe",
  "python.analysis.extraPaths": [
    "${workspaceFolder}/backend"
  ],
  "python.autoComplete.extraPaths": [
    "${workspaceFolder}/backend"
  ]
}
```

### 2. Install Project Dependencies
With the virtual environment active, install the Python requirements:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a file named `.env` in the `backend/` directory (you can copy `.env.example` as a starting template):

```bash
cp .env.example .env
```

Open the newly created `.env` file and set the configuration parameters:
*   `GEMINI_API_KEY`: Your Gemini API Key from Google AI Studio.
*   `DATABASE_URL`: Connection string to your local PostgreSQL server (e.g. `postgresql://postgres:postgres@localhost:5432/cinephile_db`).
*   `GEMINI_MODEL_NAME`: Set to `gemini-2.5-flash` for high request quotas.

### 4. Seed Relational & Vector Databases
Ensure your local PostgreSQL server is active, then run the database seeding script:

```bash
python scripts/seed_movies.py
```
This script checks if `cinephile_db` exists (creates it if missing), initializes all the database tables, loads the relational metadata cache with mock movies, runs the local embedding encoder, and indexes the vectors directly into the local Qdrant directory (`./qdrant_db`).

### 5. Launch the Application Server
Run the FastAPI application locally using Uvicorn:

```bash
uvicorn app.main:app --reload
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

Register a username, sign in, and enjoy your movie discussions!
