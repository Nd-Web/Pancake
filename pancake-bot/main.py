"""
main.py — Pancake CRM Chatbot Webhook Server

A FastAPI application that:
  1. Receives incoming customer messages from Pancake CRM via webhook.
  2. Searches the local ChromaDB knowledge base for relevant context.
  3. Uses the Groq API (free tier, Llama 3 model) to generate a helpful response.
  4. Sends the reply back to the customer through Pancake CRM's API.

Groq provides free API access to fast Llama models — sign up at
https://console.groq.com to get an API key at no cost.
"""

import os
import logging
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from chromadb import PersistentClient
from groq import Groq

# ---------------------------------------------------------------------------
# Logging setup — every incoming message and outgoing reply is logged
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pancake-bot")

# ---------------------------------------------------------------------------
# Configuration — loaded from .env file
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PANCAKE_API_KEY = os.getenv("PANCAKE_API_KEY")

# The model Groq should use. llama-3.3-70b-versatile is free-tier available
# and produces excellent conversational responses.
GROQ_MODEL = "llama-3.3-70b-versatile"

# ChromaDB settings — must match what index_knowledge.py used
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "pancake_knowledge"

# Number of relevant knowledge chunks to retrieve for context
TOP_K = 3

# ---------------------------------------------------------------------------
# Initialize FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Pancake CRM Chatbot", version="1.0.0")

# ---------------------------------------------------------------------------
# Initialize ChromaDB client and collection (lazy — created on first request)
# ---------------------------------------------------------------------------
chroma_client = None
collection = None
groq_client = None


def get_chroma_collection():
    """
    Lazily initialize the ChromaDB client and collection.
    We do this lazily so the app can start even if the DB hasn't
    been indexed yet — it will fail gracefully on queries instead.
    """
    global chroma_client, collection
    if collection is None:
        chroma_client = PersistentClient(path=CHROMA_DB_DIR)
        # get_or_create ensures we don't crash if it already exists
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' loaded — {collection.count()} chunks.")
    return collection


def get_groq_client():
    """
    Lazily initialize the Groq API client.
    """
    global groq_client
    if groq_client is None:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    return groq_client


# ---------------------------------------------------------------------------
# Core logic functions
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str, top_k: int = TOP_K) -> list[str]:
    """
    Query the ChromaDB vector store for the most relevant knowledge chunks.

    Args:
        query:  The customer's message text.
        top_k:  Number of results to return.

    Returns:
        A list of text chunks ranked by relevance.
    """
    coll = get_chroma_collection()

    # ChromaDB's query method handles embedding internally if we use
    # the same embedding function that was used during indexing.
    # Since we indexed with sentence-transformers, we need to embed
    # the query the same way.
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode(query).tolist()

    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # results["documents"] is a list of lists — grab the first (only) query's results
    chunks = results["documents"][0] if results["documents"] else []
    logger.info(f"Knowledge base search returned {len(chunks)} chunk(s) for query.")
    return chunks


def build_system_prompt(context_chunks: list[str]) -> str:
    """
    Build the system prompt that instructs the LLM how to behave.

    The prompt includes:
      - Role definition (customer service assistant)
      - Retrieved business knowledge as context
      - Strict instructions to only use provided information
      - Fallback behavior when information is unavailable
    """
    # Join the retrieved chunks into a single context block
    context_text = "\n\n".join(
        f"[Source chunk {i+1}]:\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )

    system_prompt = f"""You are a friendly and professional customer service assistant for our business. Your job is to answer customer questions accurately and helpfully.

IMPORTANT RULES:
1. ONLY use the business information provided below to answer questions. Do NOT make up or guess any information.
2. Keep your responses friendly, concise, and professional.
3. If the customer's question cannot be answered using the provided information, respond with: "Let me connect you with our team for more details on that."
4. Never invent pricing, policies, features, or any other business details.
5. If you are unsure whether the information applies, err on the side of connecting the customer with the team.

BUSINESS INFORMATION:
{context_text}"""

    return system_prompt


def generate_reply(customer_message: str) -> str:
    """
    Generate a response to the customer's message using Groq's LLM.

    Steps:
      1. Search the knowledge base for relevant context.
      2. Build a system prompt with that context.
      3. Call the Groq API to generate a response.

    Args:
        customer_message: The text of the customer's incoming message.

    Returns:
        The generated reply text.
    """
    # Step 1: Retrieve relevant knowledge
    context_chunks = search_knowledge_base(customer_message)

    # Step 2: Build the system prompt with context
    system_prompt = build_system_prompt(context_chunks)

    # Step 3: Call Groq API
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": customer_message},
        ],
        temperature=0.3,   # low temperature for factual, consistent answers
        max_tokens=500,    # keep responses concise
    )

    reply_text = response.choices[0].message.content.strip()
    return reply_text


def send_pancake_reply(conversation_id: str, reply_text: str) -> bool:
    """
    Send the generated reply back to the customer via Pancake CRM's API.

    Args:
        conversation_id: The Pancake conversation ID to reply in.
        reply_text:      The response text to send.

    Returns:
        True if the API call succeeded, False otherwise.
    """
    url = f"https://pages.fm/api/v1/conversations/{conversation_id}/messages"
    headers = {
        "Authorization": f"Bearer {PANCAKE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"message": reply_text}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            logger.info(f"✅ Reply sent to Pancake conversation {conversation_id}")
            return True
        else:
            logger.error(
                f"❌ Pancake API error: {resp.status_code} — {resp.text}"
            )
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Failed to call Pancake API: {e}")
        return False


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Returns {"status": "running"} so monitors know the service is alive.
    """
    return {"status": "running"}


@app.post("/webhook/pancake")
async def pancake_webhook(request: Request):
    """
    Main webhook endpoint that Pancake CRM calls when a new customer
    message arrives.

    Expected JSON payload from Pancake:
    {
        "conversation_id": "abc123",
        "page_id": "page_xyz",          // optional
        "message": {
            "text": "What are your prices?"
        }
    }

    Also handles Pancake's verification ping (sent as a GET-like
    challenge when first setting up the webhook).
    """
    try:
        data = await request.json()
    except Exception:
        logger.warning("⚠️  Received non-JSON request on webhook endpoint.")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "Invalid JSON payload"},
        )

    # ------------------------------------------------------------------
    # Handle Pancake verification ping
    # ------------------------------------------------------------------
    # Pancake sometimes sends a "challenge" field to verify the webhook.
    # We echo it back so Pancake confirms the endpoint is live.
    if "challenge" in data:
        logger.info("🔑 Received Pancake verification ping — echoing challenge.")
        return JSONResponse(
            content={"challenge": data["challenge"]}
        )

    # ------------------------------------------------------------------
    # Validate that we have a message with text
    # ------------------------------------------------------------------
    message_obj = data.get("message", {})
    message_text = None

    # The message field might be a dict with a "text" key, or a plain string
    if isinstance(message_obj, dict):
        message_text = message_obj.get("text")
    elif isinstance(message_obj, str):
        message_text = message_obj

    if not message_text or not message_text.strip():
        logger.info("⚠️  No message text found in payload — ignoring.")
        return {"status": "ignored"}

    # Extract conversation metadata
    conversation_id = data.get("conversation_id", "")
    page_id = data.get("page_id", "")

    logger.info(f"📥 Incoming message — conversation: {conversation_id}, "
                f"page: {page_id or 'N/A'}, text: {message_text[:80]}...")

    # ------------------------------------------------------------------
    # Generate AI reply using the knowledge base
    # ------------------------------------------------------------------
    try:
        reply_text = generate_reply(message_text)
        logger.info(f"📤 Generated reply: {reply_text[:100]}...")
    except Exception as e:
        logger.error(f"❌ Error generating reply: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)},
        )

    # ------------------------------------------------------------------
    # Send reply back through Pancake CRM
    # ------------------------------------------------------------------
    if conversation_id and PANCAKE_API_KEY:
        success = send_pancake_reply(conversation_id, reply_text)
        if not success:
            return JSONResponse(
                status_code=502,
                content={"status": "error", "detail": "Failed to send reply via Pancake API"},
            )
    else:
        logger.warning("⚠️  No conversation_id or PANCAKE_API_KEY — skipping reply delivery.")

    return {"status": "sent", "reply": reply_text}


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
