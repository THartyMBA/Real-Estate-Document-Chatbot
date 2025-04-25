# Real-Estate-Document-Chatbot
🏠📄 Real-Estate Document Q&A Chatbot
Ask questions about leases, HOA covenants, inspection reports—any real-estate PDF—without reading hundreds of pages.
Upload → Index → Chat. All on a single Streamlit page.

💡 What this app does
Upload one or more PDFs.

Splits pages into ~800-token chunks (with overlap) and embeds them using MiniLM-L6.

Builds a local FAISS vector index—no external database.

When you ask a question, the app retrieves the top-k chunks and passes them as context to a free OpenRouter LLM (default: google/gemini-2.5-pro-exp-03-25:free).

The model answers using only the retrieved excerpts.

Proof-of-concept – no user auth, persistence, or compliance controls.
Need enterprise RAG with ACLs & PII redaction? → drtomharty.com/bio

✨ Key features

Feature	Detail
Zero-cost stack	All open-source libs; free OpenRouter model; CPU-only.
Session memory	Chat remembers the full conversation for the browser session.
Multi-PDF indexing	Upload any number of docs in one go.
No long uploads	Embeddings are created client-side in RAM; nothing is stored.
Hidden context	Only the top 4 relevant text chunks are sent to the LLM—keeps tokens low.
🔑 Add your OpenRouter API key
Streamlit Cloud
⋯ → Edit secrets

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
Local dev
~/.streamlit/secrets.toml

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
—or—

bash
Copy
Edit
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
🛠️ Requirements
nginx
Copy
Edit
streamlit
langchain
faiss-cpu
sentence-transformers
PyPDF2
requests
(All CPU wheels; runs on Streamlit Cloud’s free tier.)

🚀 Quick start (local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/realestate-rag-chatbot.git
cd realestate-rag-chatbot
python -m venv venv && source venv/bin/activate          # Win: venv\Scripts\activate
pip install -r requirements.txt
streamlit run rag_realestate_chat.py
Open http://localhost:8501, upload a PDF, and ask away.

☁️ One-click deploy on Streamlit Cloud
Push the repo to GitHub.

Go to streamlit.io/cloud ➜ New app → select repo/branch.

Add OPENROUTER_API_KEY in Secrets.

Click Deploy and share your public URL.

🗂️ Repo layout
vbnet
Copy
Edit
rag_realestate_chat.py   ← single-file app
requirements.txt
README.md
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid UI

LangChain – RAG plumbing

FAISS – vector search

Sentence-Transformers – MiniLM embeddings

OpenRouter – unified LLM gateway

Upload. Ask. Understand. 🏠📑🤖
