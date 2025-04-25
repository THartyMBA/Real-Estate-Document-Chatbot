# rag_realestate_chat.py
"""
Real-Estate Document Q&A (RAG)  📄🏠
────────────────────────────────────────────────────────────────────────────
Upload one or more PDF documents (leases, HOA bylaws, inspection reports).
The app builds a **local FAISS vector index** and lets you ask questions.
Answers are generated via a free OpenRouter LLM with the retrieved excerpts
as context.

This is a lightweight proof-of-concept—perfect for demos, **not** enterprise
grade.  Need production RAG with auth, scaling, and PII controls?
Contact me → https://drtomharty.com/bio
"""
# ───────────────────────────── imports ────────────────────────────────────
import os, io, requests, tempfile
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ────────────────────── OpenRouter helper ────────────────────────────────
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
MODEL   = "google/gemini-2.5-pro-exp-03-25:free"  # free tier

def openrouter_chat(system_prompt: str, user_prompt: str, temperature=0.2):
    """Simple call to OpenRouter /chat/completions."""
    if not API_KEY:
        raise RuntimeError("Add OPENROUTER_API_KEY to env or st.secrets.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portfolio.example",   # customise
        "X-Title": "RAG-RealEstate-POC",
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ────────────────────── Embedding & Splitter ─────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# ────────────────────── PDF → text chunks ────────────────────────────────
def pdf_to_chunks(file) -> list[str]:
    reader = PdfReader(file)
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = TEXT_SPLITTER.split_text(raw_text)
    return chunks

# ────────────────────── Session init ─────────────────────────────────────
def init_session():
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("docs_uploaded", False)
    st.session_state.setdefault("chat_history", [])

# ────────────────────── UI  ───────────────────────────────────────────────
st.set_page_config(page_title="RAG Q&A for Real-Estate Docs", layout="wide")
init_session()

st.title("🏠📄 Real-Estate Document Q&A")

st.info(
    "🔔 **Demo Notice**  \n"
    "This RAG chatbot is a minimal proof-of-concept, not an enterprise solution. "
    "For production features—user auth, encryption, audit logs—[contact me](https://drtomharty.com/bio).",
    icon="💡",
)

# Sidebar – file uploader
with st.sidebar:
    st.header("📂 Upload PDFs")
    pdf_files = st.file_uploader(
        "Drag & drop one or more PDFs", type="pdf", accept_multiple_files=True
    )
    if st.button("🛠️ Build index") and pdf_files:
        with st.spinner("Reading & embedding…"):
            all_chunks = []
            for f in pdf_files:
                all_chunks.extend(pdf_to_chunks(f))
            embedder = get_embedder()
            st.session_state.vectorstore = FAISS.from_texts(all_chunks, embedder)
            st.session_state.docs_uploaded = True
        st.success("Vector index ready! Ask away ➡️")

# Main chat area
if not st.session_state.docs_uploaded:
    st.warning("Upload docs and click **Build index** to start.", icon="👈")
    st.stop()

st.subheader("Ask questions about your documents")

for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

user_query = st.chat_input("Type your question…")
if user_query:
    # Retrieve top-k context
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(user_query)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are a helpful assistant specialized in real-estate legal documents. "
        "Answer the user question **using ONLY the context provided**. "
        "If the context is insufficient, say you don't know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {user_query}"

    with st.spinner("Thinking…"):
        answer = openrouter_chat(system_prompt, user_prompt)

    # Display
    st.chat_message("user").markdown(user_query)
    st.chat_message("assistant").markdown(answer)

    # Save history
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", answer))
