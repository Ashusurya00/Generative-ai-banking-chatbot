# 🏦 AI Banking RAG Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG) chatbot** that answers banking-related queries using document-based knowledge.
Built with **Gemini API, FAISS, and Sentence Transformers**, this project enables accurate, context-aware responses from PDF and text data.

---

## 🚀 Features

* 🔍 **Semantic Search** using FAISS
* 🤖 **LLM-powered Responses** via Gemini API
* 📄 **Multi-source Document Support** (PDF + TXT)
* 🧠 **Custom RAG Pipeline (No LangChain)**
* 💬 **Interactive Chat UI** using Streamlit
* ⚡ Fast and lightweight architecture

---

## 🧠 Architecture

User Query → Embeddings → FAISS Vector Search → Relevant Context → Gemini LLM → Answer

---

## 🛠️ Tech Stack

* **LLM:** Google Gemini API
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS
* **Backend:** Python
* **Frontend:** Streamlit

---

## 📂 Project Structure

```
rag-banking-chatbot/
│
├── data/
│   ├── banking_docs.txt
│   └── banking_policy.pdf
│
├── rag_pipeline.py
├── app.py
├── requirements.txt
└── .env
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/rag-banking-chatbot.git
cd rag-banking-chatbot
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add API Key

Create a `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key
```

---

### 4️⃣ Run the App

```bash
streamlit run app.py
```

---

## 💡 Example Queries

* What is KYC?
* How to report fraud?
* What are loan eligibility criteria?

---

## 🎯 Key Highlights

* Built a **custom RAG pipeline without LangChain**
* Implemented **efficient document retrieval using FAISS**
* Designed **prompt engineering to reduce hallucinations**
* Supports **real-world document ingestion (PDF + text)**

---

## 🚀 Future Improvements

* 📂 File upload (dynamic documents)
* 🧠 Chat memory (conversation context)
* 🌐 Deployment (Streamlit Cloud / AWS)
* 📊 Evaluation metrics integration

---

## 🏆 Resume Line

**Built a production-ready RAG chatbot using Gemini, FAISS, and Sentence Transformers with PDF ingestion and context-aware response generation.**

---


