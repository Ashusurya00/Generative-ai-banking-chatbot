import streamlit as st
from rag_pipeline import ask_question

# -------- PAGE CONFIG -------- #
st.set_page_config(
    page_title="AI Banking Assistant",
    page_icon="🏦",
    layout="wide"
)

# -------- CUSTOM CSS -------- #
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .chat-bubble-user {
        background-color: #1f77b4;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        color: white;
    }
    .chat-bubble-bot {
        background-color: #2e2e2e;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------- HEADER -------- #
st.title("🏦 AI Banking Assistant")
st.caption("Ask anything about banking policies, KYC, loans, fraud, and more.")

# -------- SIDEBAR -------- #
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("This chatbot uses:")
    st.markdown("""
    - Gemini LLM  
    - FAISS Vector DB  
    - RAG Architecture  
    """)
    st.divider()
    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []

# -------- SESSION STATE -------- #
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------- CHAT DISPLAY -------- #
for role, text in st.session_state.chat:
    if role == "You":
        st.markdown(f'<div class="chat-bubble-user"><b>You:</b> {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot"><b>Assistant:</b> {text}</div>', unsafe_allow_html=True)

# -------- INPUT BOX -------- #
query = st.chat_input("Ask your banking question...")

# -------- PROCESS QUERY -------- #
if query:
    st.session_state.chat.append(("You", query))

    with st.spinner("🤖 Thinking..."):
        answer = ask_question(query)

    st.session_state.chat.append(("Bot", answer))

    st.rerun()