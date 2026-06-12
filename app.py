import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from create_database import create_vector_db

load_dotenv()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NoteGPT - Chat with your Study Material",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

.hero {
    padding: 2.5rem;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #4F46E5 0%,
        #7C3AED 50%,
        #9333EA 100%
    );
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.25);
}

.hero h1 {
    margin-bottom: 0.5rem;
}

.hero p {
    font-size: 18px;
    opacity: 0.9;
}

.metric-card {
    background: #1E293B;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #334155;
}

.source-card {
    padding: 15px;
    border-radius: 15px;
    background: #1E293B;
    border-left: 5px solid #6366F1;
    margin-bottom: 12px;
}

.sidebar-box {
    background: #1E293B;
    padding: 15px;
    border-radius: 12px;
}

.stChatMessage {
    border-radius: 16px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>📚 NoteGPT - Chat with your Study Material</h1>
    <p>
        Chat with your PDFs, Notes, Research Papers and Study Material
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="LLM",
        value="Gemini 2.5 Flash"
    )

with col2:
    st.metric(
        label="Embeddings",
        value="MPNet"
    )

with col3:
    st.metric(
        label="Retrieval",
        value="MMR Search"
    )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# FILE STORAGE
# --------------------------------------------------

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDFs / Notes",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    st.divider()

    if uploaded_files:
        st.success(
            f"{len(uploaded_files)} file(s) selected"
        )

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    build_clicked = st.button(
        "🚀 Build Knowledge Base"
    )

    if build_clicked:

        shutil.rmtree(
            UPLOAD_FOLDER,
            ignore_errors=True
        )

        os.makedirs(
            UPLOAD_FOLDER,
            exist_ok=True
        )

        for file in uploaded_files:
            with open(
                os.path.join(
                    UPLOAD_FOLDER,
                    file.name
                ),
                "wb"
            ) as f:
                f.write(file.read())

        with st.spinner(
            "Creating embeddings..."
        ):
            total_chunks = create_vector_db(
                UPLOAD_FOLDER
            )

        st.success(
            "Knowledge Base Created Successfully"
        )

        st.metric(
            "Chunks Indexed",
            total_chunks
        )

# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are CourseMate AI.

Answer only from the provided context.

If the answer is not present in the context, reply:

"I could not find this information in the uploaded documents."
            """
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
            """
        )
    ]
)

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(
            message["content"]
        )

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_query = st.chat_input(
    "Ask a question about your documents..."
)

if user_query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    try:

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 12,
                "lambda_mult": 0.5
            }
        )

        docs = retriever.invoke(
            user_query
        )

        context = "\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        final_prompt = (
            prompt.format_prompt(
                context=context,
                question=user_query
            )
            .to_messages()
        )

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):
                response = llm.invoke(
                    final_prompt
                )

            st.markdown(
                response.content
            )

            with st.expander(
                "📚 Retrieved Documents"
            ):

                for i, doc in enumerate(
                    docs,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <h4>Document {i}</h4>
                            <p>{doc.page_content[:500]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )

    except Exception as e:
        st.error(
            f"Error: {str(e)}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "CourseMate AI • RAG Powered • Gemini 2.5 Flash • ChromaDB"
)
