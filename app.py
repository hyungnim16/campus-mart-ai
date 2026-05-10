import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="MCC Campus Mart AI Guide",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&display=swap');

#MainMenu, footer, header {visibility: hidden;}

html, body, .stApp, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stBottomBlockContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
section[data-testid="stSidebar"],
.stChatFloatingInputContainer,
div[data-testid="stChatInput"],
.stChatInputContainer {
    font-family: 'Sora', sans-serif !important;
    background: #0d0d0d !important;
    background-color: #0d0d0d !important;
    color: #f0f0f0 !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 700px !important;
    background: #0d0d0d !important;
}

/* ---- INTRO ---- */
.intro-wrap {
    min-height: 10vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 1rem;
    text-align: center;
}

.intro-chip {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #aaa;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 999px;
    margin-bottom: 1.5rem;
}

.intro-heading {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.intro-heading span { color: #e05a5a; }

.intro-sub {
    font-size: 0.9rem;
    color: #777;
    max-width: 400px;
    margin: 0 auto 2rem;
    line-height: 1.7;
    font-weight: 300;
}

.feature-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}

.feat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 12px;
    color: #bbb;
    display: flex;
    align-items: center;
    gap: 7px;
}

.team-note {
    font-size: 11px;
    color: #444;
    margin-top: 1.2rem;
    letter-spacing: 0.5px;
}

/* ---- BUTTONS ---- */
div[data-testid="stButton"] button {
    background: #e05a5a !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(224,90,90,0.3) !important;
}

div[data-testid="stButton"] button:hover {
    background: #c94040 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(224,90,90,0.45) !important;
}

.back-area div[data-testid="stButton"] button {
    background: transparent !important;
    color: #555 !important;
    box-shadow: none !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.8rem !important;
    width: auto !important;
}

.back-area div[data-testid="stButton"] button:hover {
    color: #aaa !important;
    background: rgba(255,255,255,0.05) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ---- CHAT HEADER ---- */
.chat-top {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.2rem;
}

.chat-top-name {
    font-size: 1rem;
    font-weight: 700;
    color: #f0f0f0;
}

.chat-top-status {
    font-size: 11px;
    color: #666;
    display: flex;
    align-items: center;
    gap: 5px;
}

.dot {
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    display: inline-block;
}

/* ---- CHAT MESSAGES ---- */
[data-testid="stChatMessage"] {
    background: #111111 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
}

/* ---- CHAT INPUT ---- */
[data-testid="stChatInput"] textarea {
    background: #111111 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-family: 'Sora', sans-serif !important;
}

[data-testid="stChatInput"]:focus-within textarea {
    border-color: #e05a5a !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #e05a5a !important; }
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "page" not in st.session_state:
    st.session_state.page = "intro"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load knowledge base ---
@st.cache_resource
def load_knowledge_base():
    documents = []
    docs_folder = "docs"
    for file in os.listdir(docs_folder):
        filepath = os.path.join(docs_folder, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings)
    return db

# --- Prompt ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful and friendly AI assistant for MCC Campus Mart — an online e-commerce platform for Mandaue City College students, faculty, and staff.

You have two sources of knowledge:
1. The context below from MCC Campus Mart documents
2. Your own general knowledge about e-commerce, technology, and related topics

Instructions:
- If the question is about MCC Campus Mart specifically, answer using the context below
- If the question is a general topic, answer from your own knowledge
- Always keep answers short, clear and friendly — maximum 3 sentences

Context: {context}
Question: {question}
Answer:"""
)

# ================================
# INTRO PAGE
# ================================
if st.session_state.page == "intro":
    st.markdown('<div class="intro-wrap">', unsafe_allow_html=True)
    st.markdown("""
        <div class="intro-chip">MCC · Mandaue City College</div>
        <div class="intro-heading">Campus Mart<br><span>AI Guide</span></div>
        <div class="intro-sub">Your intelligent assistant for the campus marketplace — ask anything about buying, selling, donating, and trading.</div>
        <div class="feature-row">
            <div class="feat"><span>🛒</span> Buy & Sell</div>
            <div class="feat"><span>💝</span> Donate</div>
            <div class="feat"><span>🔄</span> Trade</div>
            <div class="feat"><span>🏪</span> Merchants</div>
            <div class="feat"><span>🤖</span> AI Powered</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    st.markdown("""
        <div class="team-note">Team TICNAP · JACAR — Jeneysis · Avegail · Cyrus · Angelo · Rammil</div>
    </div>
    """, unsafe_allow_html=True)

# ================================
# CHAT PAGE
# ================================
elif st.session_state.page == "chat":

    st.markdown('<div class="back-area">', unsafe_allow_html=True)
    if st.button("← Back"):
        st.session_state.page = "intro"
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="chat-top">
        <div style="font-size:1.8rem;">🎓</div>
        <div>
            <div class="chat-top-name">MCC Campus Mart AI Guide</div>
            <div class="chat-top-status"><span class="dot"></span> Online · Llama 3.3 · 70B</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading knowledge base..."):
        db = load_knowledge_base()

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt_template}
    )

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("👋 Hi! I'm your MCC Campus Mart AI Guide. Ask me anything about the system — how to buy, sell, donate, trade, or anything about the platform!")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ask me anything about MCC Campus Mart..."):
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer = qa.run(question)
            except Exception as e:
                answer = f"Error: {str(e)}"
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})