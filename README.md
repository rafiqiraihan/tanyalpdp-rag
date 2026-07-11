# TanyaLPDP – Retrieval Augmented Generation (RAG) Chatbot

## Overview

TanyaLPDP is a personal portfolio project that demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using publicly available LPDP scholarship documents as its knowledge base. This project aims to explore document search, re-ranking, and an LLM-based question-and-answer system that includes source citations. This project is not an official LPDP application or service.

## Key Higlight

* Answers questions using Retrieval-Augmented Generation (RAG)
* Combines semantic search (Chroma) and keyword search (BM25) through Hybrid Retrieval
* Improves retrieval quality using Cross-Encoder Reranking
* Provides source citations from the original LPDP documents
* Built with a modular architecture for easy maintenance and future deployment

## Project Objective

The objective of this project is to implement an end-to-end Retrieval-Augmented Generation (RAG) pipeline capable of answering questions based on LPDP scholarship documents while providing grounded responses with source citations.

This project was also developed as a personal portfolio

## Features

The features I built for this project are as follows:
* PDF ingestion
* Recursive Character Text Splitting
* Semantic Embedding using BGE-M3
* Hybrid Retrieval
* Cross-Encoder Reranking
* Source Citation
* Interactive CLI Chat Interface

## Architecture

### Indexing

<img width="300" height="434" alt="Diagram RAG TanyaLPDP drawio (2)" src="https://github.com/user-attachments/assets/931be1cc-36ad-4413-9454-853d3d243f0b" />



### Pipeline RAG

                User Question
                      │
                      ▼
              Hybrid Retrieval
          ┌───────────┴───────────┐
          │                       │
     Chroma Search           BM25 Search
          │                       │
          └───────────┬───────────┘
                      │
             Ensemble Retriever
                      │
                      ▼
            Cross-Encoder Reranker
                      │
                 Top 5 Chunks
                      │
                      ▼
                Prompt Builder
                      │
                      ▼
                   Groq LLM
                      │
                      ▼
            Answer + Source Citation

The user asks a question, which then triggers the retrieval process to find the relevant document chunks. The retrieval process uses two methods to improve accuracy by employing precise meanings and keywords to search for information stored in the database, which has already undergone indexing. Afterward, five highly relevant and precise chunks are selected. These five chunks are processed by an LLM, which operates using the Groq API and is provided with a prompt based on the project’s requirements, generating an answer along with citation sources.

## Tech Stack

| Category  | Technology                       |
| --------- | -------------------------------- |
| Language  | Python                           |
| Framework | LangChain                        |
| Vector DB | Chroma                           |
| Embedding | BAAI/bge-m3                      |
| Reranker  | BAAI/bge-reranker-base           |
| LLM       | Llama 3.3 via Groq API           |
| Retrieval | Hybrid Retrieval (Chroma + BM25) |


## Project Structure
```text
TanyaLPDP/
│
├── data/
│   ├── raw/                  # Save the original LPDP guide document as a PDF
│   └── vector_db/            # Local directory for the Chroma DB database
├── ingestion
|   └── load_data.py          # Reading and extracting text from PDFs
│
├── indexing/                 # [UPSTREAM] Initial document processing
│   ├── chunking.py           # Splitting text into several parts
│   ├── embedding.py          # BGE-M3 Embedding Model Configuration
│   ├── vector_store.py       # Storage management for Chroma DB
│   └── index.py              # Main script for running the indexing pipeline
│
├── retrieval/                # [CENTER] Document search engine
│   └── searcher.py           # Combination of BM25 + Chroma Vector + Reranker (Hybrid Search)
│
├── generation/               # [DOWNSTREAM] Answer Compiler
│   ├── prompt_builder.py     # Llama 3.3 Strict Instruction Template (Guardrail)
│   └── generator.py          # Initialization of the Groq API & RAG Chain Pipeline Builder (LCEL)
│
├── .env                      # Storing a Secret API Key (Groq API Key)
├── .gitignore                # Exclude venv/, .env, and database/ from GitHub
├── main.py                   # Key Components of a Terminal CLI-Based Chatbot Application
├── requirements.txt          # List of all Python libraries that must be installed
└── README.md                 # Project Guide Documentation
```

## Installation

Follow these steps to set up and run the TanyaLPDP Chatbot on your local machine:

### 1. Clone the Repository
Open your terminal and clone this repository:
```bash
git clone https://github.com/rafiqiraihan/TanyaLPDP.git
cd TanyaLPDP
```

### 2. Create and Activate a Virtual Environment
It highly recomended to use Python 3.10.11 or newer.
* Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```
* Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required Python packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables(.env)
Create a new file named .env in the root directory of the project (TanyaLPDP/), and add your Groq API Key:
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 5. Run the Indexing Pipeline (One-Time Setup)
Place your official guide PDF file into the data/raw/ directory, then run the indexing script to parse, chunk, and save the data into the local Chroma vector database:
```bash
python indexing/index.py
```

### 6. Launch the Chatbot (CLI Mode)
Once the database has been built start main conductor script to interact with your virtual assistant directly inside your terminal:
```bash
python main.py
```

### Quick Check before saving:
Make sure your file structure exactly matches the references (`indexing/index.py` and `main.py`). If you haven't generated your `requirements.txt` yet, remember to do it inside your activated `(venv)` using:

```bash
pip freeze > requirements.txt
```

## Demo

nanti ditambah di github

## Future Improvements
* FastAPI REST API
* Streamlit Web Interface
* Docker Deployment
* Retrieval Evaluation Framework
* Query Expansion
* HyDE Retrieval
* Conversation Memory
