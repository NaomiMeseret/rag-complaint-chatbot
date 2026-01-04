# 🤖 Intelligent Complaint Analysis for Financial Services

> A **RAG-Powered Chatbot** that transforms customer complaints into actionable insights for financial services teams.

## 📋 Overview

CrediTrust Financial receives thousands of customer complaints monthly across multiple financial products. This project builds an **AI-powered chatbot** that enables Product Managers, Support Teams, and Compliance Officers to quickly identify trends, extract insights, and answer questions about customer feedback—turning raw complaint data into strategic business intelligence.

### 🎯 Key Objectives

- ⚡ **Reduce analysis time** from days to minutes for identifying complaint trends
- 👥 **Empower non-technical teams** to query complaint data without data analysts
- 🔮 **Enable proactive problem-solving** by identifying issues before they escalate

---

## 💡 Project Idea

This system uses **Retrieval-Augmented Generation (RAG)** to answer questions about customer complaints. Here's how it works:

1. **📊 Data Processing**: Clean and preprocess complaint narratives from the CFPB dataset
2. **✂️ Text Chunking**: Break long complaint narratives into smaller, searchable chunks
3. **🧠 Embedding Generation**: Convert text chunks into vector embeddings using semantic models
4. **🔍 Vector Search**: Store embeddings in a vector database (ChromaDB) for fast semantic search
5. **💬 Question Answering**: When users ask questions, the system:
   - Retrieves the most relevant complaint chunks using semantic similarity
   - Passes retrieved context to a language model
   - Generates concise, evidence-backed answers

### 📦 Products Analyzed

- 💳 Credit Cards
- 💰 Personal Loans
- 🏦 Savings Accounts
- 💸 Money Transfers

---

## 🛠️ Technology Stack

- **Python 3.8+**
- **Pandas & NumPy** - Data processing and analysis
- **Sentence Transformers** - Text embeddings (`all-MiniLM-L6-v2`)
- **ChromaDB** - Vector database for semantic search
- **scikit-learn** - Stratified sampling
- **Jupyter Notebooks** - EDA and development

---

## 📁 Project Structure

```
rag-complaint-chatbot/
├── 📂 data/
│   ├── raw/                    # Raw CFPB complaint dataset
│   └── processed/              # Cleaned and filtered data
├── 📂 vector_store/            # ChromaDB persistent vector store
├── 📂 notebooks/
│   ├── 01_task1_eda_preprocessing.ipynb
│   └── 02_task2_chunk_embed_vectorstore.ipynb
├── 📂 src/                     # Source code (for future tasks)
├── 📂 tests/                   # Test scripts
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd rag-complaint-chatbot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download the CFPB complaint dataset**
   - Place the dataset file in `data/raw/complaints.csv`
   - The dataset should contain consumer complaint narratives from CFPB

---

## 📝 Tasks & Deliverables

### ✅ Task 1: Exploratory Data Analysis & Preprocessing

- Analyze complaint data structure and distributions
- Filter complaints for target product categories
- Clean and normalize complaint narratives
- Generate visualizations and summary statistics

**Output**: `data/filtered_complaints.csv`

### ✅ Task 2: Text Chunking, Embedding & Vector Store Indexing

- Create stratified sample (10K-15K complaints)
- Implement text chunking strategy (500 chars, 50 overlap)
- Generate embeddings using MiniLM-L6-v2
- Index embeddings and metadata in ChromaDB

**Output**: Persistent vector store in `vector_store/`

### 🔄 Task 3-4: RAG Pipeline & User Interface (Future)

- Build RAG query pipeline with LLM integration
- Develop Gradio/Streamlit interface
- Evaluate and optimize retrieval performance

---

## 🔬 Key Design Decisions

### Text Chunking

- **Chunk Size**: 500 characters
- **Overlap**: 50 characters
- **Rationale**: Balances context preservation with embedding quality

### Embedding Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Rationale**: Fast, accurate, and optimized for semantic search tasks

### Vector Database

- **Database**: ChromaDB
- **Rationale**: Lightweight, persistent, and easy to integrate with Python

---

## 👥 Target Users

- **Product Managers** - Identify feature issues and user pain points
- **Customer Support Teams** - Quickly understand common complaint patterns
- **Compliance & Risk Teams** - Monitor regulatory and fraud signals
- **Executives** - Gain visibility into emerging customer issues

---

## 📊 Expected Outcomes

By completing this project, teams will be able to:

- Ask natural language questions like _"Why are customers unhappy with credit cards?"_
- Receive synthesized answers backed by relevant complaint narratives
- Filter and compare issues across different financial products
- Identify trends and patterns in real-time

---

**Dataset**: Consumer Financial Protection Bureau (CFPB) Complaint Database

- **Embedding Model**: sentence-transformers by SBERT
- **Vector Database**: ChromaDB

---

**Built with ❤️ for better customer insights**
