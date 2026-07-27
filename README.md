# TanyaLPDP – Retrieval Augmented Generation (RAG) Chatbot

> An end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers questions about LPDP scholarships using official documents, featuring hybrid retrieval, Cross-Encoder reranking, source citation, and a FastAPI-based REST API.

## Overview

TanyaLPDP is an end-to-end Retrieval-Augmented Generation (RAG) application that answers questions about LPDP scholarship programs using official documents as its knowledge base. The system combines hybrid retrieval, Cross-Encoder reranking, and LLM-based generation to provide grounded responses with source citations. It is designed as a portfolio project to demonstrate AI Engineering practices, including modular architecture, API serving with FastAPI, and containerization with Docker.

## Demo

<img width="1301" height="910" alt="Screenshot 2026-07-11 193045" src="https://github.com/user-attachments/assets/1be7262e-d753-4d23-8d56-b5187988f5cb" />

## Key Highlights

* Answers questions using official LPDP scholarship documents
* Hybrid Retrieval (BM25 + Vector Search)
* Cross-Encoder Reranking for improved retrieval precision
* Grounded responses with source citations
* FastAPI REST API
* Dockerized application
* Modular AI pipeline designed for maintainability

## Why This Project

Official LPDP scholarship guides contain hundreds of pages, making manual information retrieval time-consuming.

This project demonstrates how Retrieval-Augmented Generation (RAG) can provide grounded answers with citations while reducing hallucinations through retrieval, reranking, and prompt engineering.

## Engineering Decisions

### Why Hybrid Retrieval?

Dense retrieval captures semantic similarity, while BM25 excels at exact keyword matching. Combining both improves retrieval recall across different query types.

---

### Why Cross-Encoder Reranking?

The initial retrieval stage prioritizes recall by returning multiple candidate chunks. A Cross-Encoder reranks these candidates to improve precision before passing them to the LLM.

---

### Why Source Citation?

Every generated answer includes document references so users can verify information directly from the original LPDP documents, reducing hallucination risk.

---

### Why FastAPI?

FastAPI provides lightweight, high-performance REST APIs for serving the RAG pipeline and simplifies integration with external applications.

---

### Why Docker?

Containerization ensures reproducible environments and simplifies deployment across different machines.

## Features

The features I built for this project are as follows:
* PDF ingestion
* Recursive Character Text Splitting
* Semantic Embedding using BGE-M3
* Hybrid Retrieval
* Cross-Encoder Reranking
* Source Citation
* RESTful API using FastAPI

## Architecture

### Indexing

<img width="300" height="434" alt="Diagram RAG TanyaLPDP drawio (2)" src="https://github.com/user-attachments/assets/931be1cc-36ad-4413-9454-853d3d243f0b" />



### Pipeline RAG

<img width="453" height="623" alt="Diagram RAG TanyaLPDP drawio" src="https://github.com/user-attachments/assets/c901d5d3-1f7b-49c9-bf8e-901e1921af15" />


1. User sends a question via FastAPI.

2. Hybrid Retrieval retrieves candidate chunks.

3. Cross-Encoder reranks the candidates.

4. Top-ranked chunks are passed to the LLM.

5. The LLM generates a grounded answer with source citations.

## Tech Stack

| Category              | Technology                       |
| --------------------- | -------------------------------- |
| Language              | Python                           |
| Backend               | FastAPI                          |
| AI Framework          | LangChain                        |
| Vector Database       | ChromaDB                         |
| Embedding             | BAAI/bge-m3                      |
| Retrieval             | Hybrid Retrieval (Chroma + BM25) |
| Reranker              | BAAI/bge-reranker-base           |
| LLM                   | Llama 3.3 via Groq API           |
| Containerization      | Docker                           |


## Project Structure
```text
TanyaLPDP/
│
├── api/                       # FastAPI application, request/response schemas, and API endpoints
│   ├── main.py 
│   └── schemas.py 
│ 
├── config/                   # Centralized project configuration
│   └── config.py 
│ 
├── data/
│   ├── raw/                  # Original LPDP PDF documents
│   └── vector_db/            # Persisted Chroma vector database
│
├── ingestion                 # PDF loading and text extraction
|   └── load_data.py
│
├── indexing/                 # Document chunking, embedding, and indexing pipeline
│   ├── chunking.py          
│   ├── embedding.py          
│   ├── vector_store.py       
│   └── index.py          
│
├── retrieval/                # Hybrid retrieval and reranking logic
│   └── searcher.py           
│
├── generation/               # Prompt engineering and LLM integration
│   ├── prompt_builder.py     
│   └── generator.py          
│
├── pipeline/                 # End-to-end RAG orchestration
│   └── pipeline.py           
│
├── .dockerignore             
├── Dockerfile                 
├── .env                      
├── .gitignore                
├── main.py                   
├── requirements-docker.txt   
├── requirements.txt          
└── README.md                 
```

## Installation

Follow these steps to set up and run TanyaLPDP locally:

### 1. Clone the Repository
Open your terminal and clone this repository:
```bash
git clone https://github.com/rafiqiraihan/TanyaLPDP.git
cd TanyaLPDP
```

### 2. Create and Activate a Virtual Environment
It is highly recomended to use Python 3.10.11 or newer.
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
Note: The indexing pipeline only needs to be executed once, or whenever the source documents are updated.

### 6. Run the FastAPI Application
Once the indexing process has completed, start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

## Docker Installation

Build the Docker image:

```bash
docker build -t tanyalpdp .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env tanyalpdp
```

The API will be available at:

```text
http://localhost:8000
```

Once the container is running, open:

```text
http://localhost:8000/docs
```
## Future Improvements
* Retrieval Evaluation Framework
* Query Expansion
* HyDE Retrieval
* Conversation Memory
* Streamlit Web Interface
