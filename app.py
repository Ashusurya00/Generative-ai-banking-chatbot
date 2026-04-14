from datetime import datetime
import streamlit as st
from rag_pipeline import ask_question, get_knowledge_base_stats
st.set_page_config(
    page_title="Banking AI Copilot",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded",
)
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: rgba(10, 19, 34, 0.78);
            --surface-strong: rgba(13, 26, 45, 0.96);
            --surface-soft: rgba(21, 37, 60, 0.86);
            --text: #f5f7fb;
            --muted: #9eb3cc;
            --line: rgba(173, 201, 234, 0.14);
            --primary: #89aefc;
            --primary-deep: #c2d6ff;
            --accent: #76e7bd;
            --gold: #f2c879;
            --shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(137, 174, 252, 0.18), transparent 28%),
                radial-gradient(circle at 88% 14%, rgba(118, 231, 189, 0.12), transparent 22%),
                radial-gradient(circle at 50% 100%, rgba(242, 200, 121, 0.08), transparent 30%),
                linear-gradient(135deg, #040914 0%, #091221 42%, #10203a 100%);
            color: var(--text);
        }
        .main .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(137, 174, 252, 0.12), transparent 28%),
                linear-gradient(180deg, #060f1d 0%, #0d1728 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        [data-testid="stSidebar"] * {
            color: #eef4ff !important;
        }
        .hero-card {
            padding: 2.2rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(137, 174, 252, 0.16), transparent 30%),
                linear-gradient(135deg, rgba(16, 29, 51, 0.98), rgba(7, 15, 29, 0.94));
            border: 1px solid rgba(194, 214, 255, 0.16);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -10% -45% auto;
            width: 280px;
            height: 280px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(242, 200, 121, 0.14), transparent 60%);
            pointer-events: none;
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            background: rgba(242, 200, 121, 0.10);
            color: var(--gold);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid rgba(242, 200, 121, 0.18);
        }
        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1.02;
            margin: 1rem 0 0.8rem 0;
            color: var(--text);
            max-width: 760px;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            line-height: 1.75;
            color: var(--muted);
            max-width: 860px;
        }
        .metric-card, .info-card, .source-card {
            background: linear-gradient(180deg, rgba(17, 32, 56, 0.96), rgba(10, 21, 37, 0.92));
            backdrop-filter: blur(12px);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.1rem 1rem;
            box-shadow: var(--shadow);
        }
        .metric-card {
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--primary), var(--accent), var(--gold));
        }
        .metric-label {
            font-size: 0.82rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--text);
            margin-top: 0.25rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
            color: var(--text);
        }
        .chat-shell {
            background: linear-gradient(180deg, rgba(12, 23, 40, 0.92), rgba(8, 16, 29, 0.9));
            border: 1px solid rgba(194, 214, 255, 0.10);
            border-radius: 24px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }
        .trust-pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }
        .trust-pill.primary {
            background: rgba(137, 174, 252, 0.12);
            color: var(--primary-deep);
        }
        .trust-pill.success {
            background: rgba(118, 231, 189, 0.10);
            color: var(--accent);
        }
        .trust-pill.warning {
            background: rgba(242, 200, 121, 0.10);
            color: var(--gold);
        }
        .stChatMessage {
            background: linear-gradient(180deg, rgba(17, 31, 53, 0.92), rgba(10, 20, 36, 0.9));
            border: 1px solid rgba(194, 214, 255, 0.10);
            border-radius: 18px;
        }
        [data-testid="stChatMessageContent"] {
            color: var(--text);
        }
        .stAlert {
            background: rgba(15, 27, 46, 0.96);
            color: var(--text);
            border: 1px solid rgba(194, 214, 255, 0.12);
        }
        .stButton > button {
            background: linear-gradient(135deg, rgba(18, 34, 57, 0.98), rgba(11, 22, 39, 0.95));
            color: var(--text);
            border: 1px solid rgba(194, 214, 255, 0.14);
            border-radius: 14px;
            padding: 0.75rem 1rem;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            color: #ffffff;
            border-color: rgba(242, 200, 121, 0.42);
            transform: translateY(-1px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
        }
        .stChatInputContainer > div {
            background: rgba(9, 18, 33, 0.96);
            border: 1px solid rgba(194, 214, 255, 0.14);
        }
        .stChatInputContainer textarea {
            color: #f5f7fb !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #f5f7fb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <span class="hero-kicker">Enterprise GenAI Banking Demo</span>
            <div class="hero-title">Banking AI Copilot</div>
            <div class="hero-subtitle">
                A polished Gemini-powered RAG assistant for policy lookup, customer-service operations,
                compliance Q&amp;A, and risk-aware response drafting. Built to feel closer to an internal
                banking copilot than a classroom chatbot.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_sidebar(stats: dict) -> None:
    with st.sidebar:
        st.markdown("## Control Center")
        st.caption("Enterprise-style operational overview")
        st.divider()
        st.markdown(
            """
            LLM: Gemini 2.5 Flash  
            Retrieval: FAISS + Sentence Transformers  
            Pattern: Retrieval-Augmented Generation  
            Domain: Banking policy and service guidance
            """
        )
        st.divider()
        st.metric("Knowledge Assets", stats["documents"])
        st.metric("Indexed Chunks", stats["chunks"])
        st.metric("Policy Pages", stats["pdf_pages"])
        st.divider()
        st.markdown("### Suggested Use Cases")
        st.markdown(
            """
            - KYC and onboarding queries
            - Fraud reporting workflows
            - Account and card guidance
            - Loan and eligibility policy lookup
            """
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
def render_suggested_prompts() -> str | None:
    st.markdown('<div class="section-title">Launch With A Strong Prompt</div>', unsafe_allow_html=True)
    prompts = [
        "Explain the KYC process and why it matters for banking customers.",
        "What should a customer do immediately after detecting fraudulent activity?",
        "Summarize the factors that affect loan eligibility.",
        "Compare savings accounts and credit cards using the available knowledge base.",
    ]
    selected_prompt = None
    columns = st.columns(2)
    for index, prompt in enumerate(prompts):
        with columns[index % 2]:
            if st.button(prompt, key=f"prompt_{index}", use_container_width=True):
                selected_prompt = prompt
    return selected_prompt
def render_sources(sources: list[dict]) -> None:
    st.markdown('<div class="section-title">Retrieved Evidence</div>', unsafe_allow_html=True)
    if not sources:
        st.info("No source context was returned for this answer.")
        return
    for item in sources:
        source_header = f"{item['source_name']} - {item['source_type']}"
        if item["page"]:
            source_header += f" - Page {item['page']}"
        st.markdown(
            f"""
            <div class="source-card">
                <strong>{source_header}</strong><br/>
                <span style="color:#9eb3cc;">Similarity score: {item['score']:.3f}</span>
                <div style="margin-top:0.75rem; color:#f5f7fb; line-height:1.7;">{item['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
inject_styles()
if "messages" not in st.session_state:
    st.session_state.messages = []
stats = get_knowledge_base_stats()
render_sidebar(stats)
render_hero()
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    render_metric_card("Gemini Model", "2.5 Flash")
with metric_col2:
    render_metric_card("Response Mode", "Grounded RAG")
with metric_col3:
    render_metric_card("Interface", "Enterprise UI")
overview_col, trust_col = st.columns([1.6, 1])
with overview_col:
    st.markdown(
        """
        <div class="info-card">
            <div class="section-title">What Makes This Feel More Production-Ready</div>
            Answers are generated from retrieved banking content, carry source transparency, and surface a
            confidence signal so the experience feels closer to a knowledge copilot used by operations or
            customer-support teams.
        </div>
        """,
        unsafe_allow_html=True,
    )
with trust_col:
    st.markdown(
        """
        <div class="info-card">
            <div class="section-title">Trust Signals</div>
            <span class="trust-pill primary">Grounded answers</span>
            <span class="trust-pill success">Source evidence</span>
            <span class="trust-pill warning">Confidence visibility</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
suggested_prompt = render_suggested_prompts()
if suggested_prompt:
    st.session_state.pending_prompt = suggested_prompt
chat_col, source_col = st.columns([1.65, 1], gap="large")
with chat_col:
    st.markdown('<div class="section-title">Analyst Conversation</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.info(
            "Start with a policy, compliance, account, loan, or fraud-related question to see the upgraded Gemini RAG flow."
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(
                    f"Confidence: {message.get('confidence_label', 'N/A')} ({message.get('confidence_score', 0.0):.3f})"
                )
    st.markdown("</div>", unsafe_allow_html=True)
with source_col:
    latest_assistant = next(
        (message for message in reversed(st.session_state.messages) if message["role"] == "assistant"),
        None,
    )
    render_sources(latest_assistant["sources"] if latest_assistant else [])
query = st.chat_input("Ask a banking, compliance, fraud, account, or loan question...")
if not query and st.session_state.get("pending_prompt"):
    query = st.session_state.pop("pending_prompt")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    chat_history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1]
    ]
    try:
        with st.spinner("Analyzing the banking knowledge base with Gemini..."):
            result = ask_question(query, chat_history=chat_history)
    except Exception as exc:
        result = {
            "answer": f"Unable to complete the request right now. Details: {exc}",
            "sources": [],
            "confidence_label": "Low",
            "confidence_score": 0.0,
        }
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence_label": result["confidence_label"],
            "confidence_score": result["confidence_score"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    st.rerun()
