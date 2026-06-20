import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, chat, memory, movies, analytics
from app.core.database import init_db_schema, engine
from app.services.cache import cache_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="CinephileGPT API",
    description="Conversational AI assistant whose entire personality revolves around movies.",
    version="1.0.0"
)

# CORS setup to allow local frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to actual origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    try:
        logger.info("Attempting to connect and initialize/migrate PostgreSQL tables...")
        init_db_schema()
        logger.info("Relational database tables checked/created/migrated successfully.")
    except Exception as e:
        logger.error(f"Error during PostgreSQL schema initialization: {e}")
        logger.warning("FastAPI started successfully, but SQL operations will fail until PostgreSQL is running and correct connection details are provided in .env.")

    # Initialize Redis caching connection pool
    await cache_service.connect()

@app.on_event("shutdown")
async def on_shutdown():
    # Disconnect from Redis pool gracefully
    await cache_service.disconnect()

# Register routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(movies.router)
app.include_router(analytics.router)

import os
from fastapi.responses import HTMLResponse

@app.get("/")
def read_root():
    static_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(static_path):
        with open(static_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
            
    return HTMLResponse(content="""
    <html>
        <head><title>CinephileGPT API</title></head>
        <body style="font-family: sans-serif; background: #111; color: #eee; text-align: center; padding: 100px 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: #1e1e1e; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); border: 1px solid #333;">
                <h1 style="color: #e50914; margin-top: 0;">CinephileGPT API is Online 🎬</h1>
                <p style="font-size: 1.1em; line-height: 1.6;">Your local FastAPI backend is active and ready to discuss cinema, query database recommendations, and deflect coding questions!</p>
                <p style="color: #999; font-size: 0.9em; margin-top: 30px;">Create and place <code>index.html</code> in <code>backend/static/</code> to load the interactive dashboard.</p>
            </div>
        </body>
    </html>
    """)
