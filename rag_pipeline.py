import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# -------- LOAD ENV -------- #
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# -------- LOAD DOCUMENTS -------- #
def load_documents():
    texts = []

    # TXT FILE
    if os.path.exists("data/banking_docs.txt"):
        with open("data/banking_docs.txt", "r", encoding="utf-8") as f:
            texts.append(f.read())

    # PDF FILE
    if os.path.exists("data/banking_policy.pdf"):
        reader = PdfReader("data/banking_policy.pdf")
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

    return texts

documents = load_documents()

# -------- SPLIT TEXT -------- #
def split_text(texts, chunk_size=500):
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks

chunks = split_text(documents)

# -------- EMBEDDINGS -------- #
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------- CHROMA DB -------- #
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="banking_docs"
)

# Add documents to Chroma
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[str(i)]
    )

# -------- RETRIEVAL -------- #
def retrieve(query, k=3):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results["documents"][0]

# -------- MAIN FUNCTION -------- #
def ask_question(query):
    docs = retrieve(query)

    context = "\n".join(docs)

    prompt = f"""
    You are a professional banking assistant.

    Use ONLY the context below to answer.
    If the answer is not in the context, say:
    "I don't have enough information."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = model.generate_content(prompt)
    return response.text
