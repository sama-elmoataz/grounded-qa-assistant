NovaDesk RAG Assistant

A Retrieval-Augmented Generation (RAG) application built with Streamlit for answering questions from the NovaDesk Company Information Guide.

The assistant retrieves relevant sections from the company PDF, reranks the retrieved chunks, sends only the strongest context to the LLM, and returns a grounded answer with page-level references and supporting excerpts.

NovaDesk is a fictional company knowledge base created for demonstration and portfolio purposes.

Overview

NovaDesk Assistant is designed to demonstrate a complete end-to-end RAG workflow with a polished user experience.

Users can ask questions about:

Plans and pricing

Billing and failed payments

Support and SLAs

Security and data retention

Integrations and API limits

Troubleshooting and onboarding

The application is grounded only in the uploaded NovaDesk company guide and is instructed not to invent information when the answer is not available in the source document.

Key Features

PDF-based company knowledge assistant

BGE sentence embeddings

FAISS vector search

Top-10 document retrieval

FlashRank reranking to Top 4 results

OpenRouter-compatible LLM generation

Page-level source references

Supporting retrieved excerpts

Multi-turn chat history

Suggested questions

Category-based navigation

Copy-answer button

Export-chat functionality

Source page count badges

Custom light-luxury Streamlit interface

Cached RAG resources for faster repeated queries

RAG Pipeline

NovaDesk PDF
     ↓
PyPDFLoader
     ↓
Text Chunking
     ↓
BGE Embeddings
     ↓
FAISS Vector Store
     ↓
Top 10 Retrieval
     ↓
FlashRank Reranking
     ↓
Top 4 Documents
     ↓
Prompt + Retrieved Context
     ↓
LLM via OpenRouter
     ↓
Grounded Answer
     ↓
Page References + Supporting Context

Tech Stack

Component

Technology

Frontend

Streamlit

RAG orchestration

LangChain

PDF loading

PyPDFLoader

Embeddings

BAAI/bge-small-en-v1.5

Vector database

FAISS

Retrieval

Semantic similarity search

Reranking

FlashRank

LLM interface

ChatOpenAI via OpenRouter

Prompting

LangChain ChatPromptTemplate

Output parsing

StrOutputParser

Project Structure

grounded-qa-assistant/
│
├── app.py
│
├── data/
│   ├── novadesk_company_information.pdf
│   └── vectorstore/
│       ├── index.faiss
│       └── index.pkl
│
├── src/
│   ├── config.py
│   ├── generation.py
│   ├── ingestion.py
│   ├── reranking.py
│   └── retrieval.py
│
├── requirements.txt
├── README.md
└── .gitignore

How It Works

1. Document ingestion

The NovaDesk PDF is loaded and split into overlapping text chunks.

Current chunk configuration:

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

2. Embeddings

Each chunk is converted into a vector representation using:

BAAI/bge-small-en-v1.5

3. Vector storage

The embeddings are stored in a FAISS index for efficient semantic retrieval.

4. Retrieval

For each user question, the application retrieves the Top 10 most relevant chunks:

RETRIEVAL_K = 10

5. Reranking

FlashRank reranks the retrieved documents and keeps the strongest 4:

RERANK_TOP_N = 4

6. Generation

The reranked documents are formatted with their source page numbers and inserted into the prompt.

The LLM is explicitly instructed to:

use only retrieved NovaDesk context

avoid outside knowledge

avoid inventing policies, prices, dates, limits, or exceptions

return a fallback response when the information is not present

7. Sources

The interface displays the PDF page numbers used for the response and allows the user to inspect the supporting retrieved text.

Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/grounded-qa-assistant.git
cd grounded-qa-assistant

2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

Environment Variables

Create a local .env file in the project root:

OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=your_openrouter_model_name

For example:

OPENROUTER_API_KEY=your_key_here
LLM_MODEL=openai/gpt-4o-mini

Do not commit .env to GitHub.

The repository .gitignore should contain:

.env
.env.*
.streamlit/secrets.toml

Running Locally

Start the Streamlit application:

streamlit run app.py

Then open the local URL shown in the terminal, usually:

http://localhost:8501

Rebuilding the Vector Store

If the PDF changes, rebuild the FAISS index using the ingestion workflow before running the application again.

The vector store is stored at:

data/vectorstore/

The application currently expects:

index.faiss
index.pkl

Keeping the prebuilt index in the repository makes the hosted demo faster because the embeddings do not need to be regenerated during every cold start.

Streamlit Community Cloud Deployment

The application can be deployed directly from GitHub using Streamlit Community Cloud.

Deployment configuration

Use:

Repository: your GitHub repository
Branch: main
Main file: app.py

Streamlit Secrets

Do not upload your local .env file.

Instead, add the following values under:

Streamlit App → Settings → Secrets

Use TOML format:

OPENROUTER_API_KEY = "your_real_key"
LLM_MODEL = "your_model_name"

The application reads these values through environment variables.

Security

The repository is designed so credentials remain outside source control.

Never commit:

.env
.streamlit/secrets.toml
API keys
access tokens
passwords

Recommended checks before making the repository public:

git check-ignore .env

git log --all -- .env

git grep -n -i -E "sk-or-|api_key|password|secret|token"

Environment-variable names appearing in the code are normal. Actual secret values should never appear in the repository.

For a public demo, use a dedicated API key with appropriate usage or spending limits rather than a personal development key.

Example Questions

Try asking:

Compare NovaDesk Starter, Growth, and Enterprise plans, including price, minimum seats, and key inclusions.

What is the NovaDesk failed card payment timeline from grace period to possible suspension?

What are NovaDesk API request-per-minute limits for Starter, Growth, and Enterprise?

What are NovaDesk audit log retention periods for Growth and Enterprise?

Grounding Strategy

The generation prompt is intentionally restrictive.

If the retrieved context does not contain enough information, the assistant responds with:

The information is not stated in the NovaDesk Company Information Guide.

This reduces unsupported answers and keeps responses aligned with the source document.

UI Highlights

The Streamlit interface includes:

custom NovaDesk branding

light-luxury visual theme

knowledge category navigation

quick-question shortcuts

multi-turn conversation view

retrieved page badges

expandable supporting context

copy-answer control

downloadable conversation export

loading state while retrieval and generation are running

Future Improvements

Potential extensions include:

hybrid BM25 + vector retrieval

metadata filtering

query rewriting

conversation-aware retrieval

retrieval evaluation metrics

automated RAG evaluation datasets

answer faithfulness scoring

citation-level highlighting

user authentication

document upload and dynamic indexing

multiple knowledge-base support

admin analytics for common questions

Purpose

This project demonstrates practical implementation of:

Retrieval-Augmented Generation

semantic search

embeddings

vector databases

reranking

prompt grounding

LLM integration

Streamlit deployment

secret management

production-style RAG UX

It is intended as a portfolio and demonstration project rather than a production company knowledge system.

License

This project is provided for educational and portfolio use.