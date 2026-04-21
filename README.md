# HR Policy Assistant
An AI-powered HR assistant built using LangGraph and RAG to answer company policy queries.

## Problem Statement
Employees frequently ask HR about leave, payroll, and policies. This project builds an AI assistant to answer these queries 24/7.

## Features
- LangGraph-based agent
- Retrieval-Augmented Generation (ChromaDB)
- Multi-turn memory using thread_id
- Tool support (date/time)
- Self-evaluation loop
- Streamlit UI

## Tech Stack
- Python
- LangGraph
- Groq LLM
- ChromaDB
- Streamlit

## Run Locally

```bash
pip install -r requirements.txt
streamlit run capstone_streamlit.py
```

## Screenshots

### Chat Interface
![Chat UI](screenshots/screenshot1.png)

### Response Example
![Response](screenshots/screenshot2.png)

## ⚙️ How It Works
1. User query is processed through LangGraph
2. Router decides: retrieve / tool / memory
3. Relevant documents fetched from ChromaDB
4. LLM generates grounded response
5. Self-evaluation ensures answer quality

## Author
Name: Preetam Mondal 
Roll No.: 23051768 
Program: B.Tech(CSE)[2023-27] 
Batch: Agentic-AI 

GitHub: https://github.com/mpreetam07/hr-policy-agent