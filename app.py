import streamlit as st
import os
import time

# ==========================================================
# IMPORT RAG PIPELINE (DO NOT MODIFY rag.py)
# ==========================================================
from rag import index_documents, get_answer, vectorstore

# ==========================================================
# PAGE CONFIGURATION (ChatGPT Style)
# ==========================================================
st.set_page_config(
    page_title="ChatGPT · RAG Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CHATGPT CSS THEME
# ==========================================================
st.markdown("""
<style>
/* Base Dark Theme (ChatGPT Charcoal) */
.stApp {
    background-color: #212121;
    color: #ececec;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Hide default Streamlit elements */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer, .stDeployButton {
    display: none !important;
}

/* Sidebar styling like ChatGPT */
section[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2f2f2f !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2f2f2f !important;
}

/* Chat container width constraint */
.main .block-container {
    max-width: 820px !important;
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    padding: 1.2rem 1.4rem !important;
    border-radius: 18px !important;
    margin-bottom: 1rem !important;
    border: none !important;
    line-height: 1.6 !important;
}

/* User Message (ChatGPT Dark Grey Bubble on the right) */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background-color: #2f2f2f !important;
    color: #f3f3f3 !important;
    border-radius: 20px 20px 4px 20px !important;
    max-width: 82% !important;
    margin-left: auto !important;
}

/* Assistant Message */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background-color: transparent !important;
    color: #ececec !important;
    border-radius: 12px !important;
    width: 100% !important;
}

/* Streamlit Chat Input Capsule (ChatGPT Floating Bar) */
div[data-testid="stChatInput"] {
    border-radius: 28px !important;
    background-color: #2f2f2f !important;
    border: 1px solid #3f3f3f !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: #676767 !important;
    box-shadow: 0 4px 28px rgba(0, 0, 0, 0.5) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #f1f1f1 !important;
    background-color: transparent !important;
    font-size: 0.95rem !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #8e8e8e !important;
}

div[data-testid="stChatInput"] button {
    border-radius: 50% !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    width: 32px !important;
    height: 32px !important;
}

div[data-testid="stChatInput"] button:disabled {
    background-color: #424242 !important;
    color: #737373 !important;
}

/* Sidebar New Chat Button */
.new-chat-btn button {
    background-color: #212121 !important;
    border: 1px solid #333333 !important;
    color: #ececec !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.new-chat-btn button:hover {
    background-color: #2f2f2f !important;
    border-color: #4a4a4a !important;
}

/* Expanders in Sidebar */
div[data-testid="stExpander"] {
    background-color: #212121 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
}

/* Badges and text */
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    background-color: #2f2f2f;
    border: 1px solid #444444;
    font-size: 0.75rem;
    color: #d1d5db;
    font-weight: 500;
}

.chunk-badge {
    color: #10a37f;
    font-weight: 600;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #424242;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #555555;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Stream typewriter response generator
def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)

# ==========================================================
# SIDEBAR (ChatGPT Style)
# ==========================================================
with st.sidebar:
    # Top Branding & New Chat Button
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <div style="width: 34px; height: 34px; border-radius: 10px; background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%); display: flex; align-items: center; justify-content: center; font-size: 18px; color: white;">
                ✨
            </div>
            <div>
                <div style="font-weight: 700; font-size: 16px; color: #ffffff; letter-spacing: -0.3px;">RAG Assistant</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # + New Chat Button
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", key="new_chat_btn", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Knowledge Base / Document Attachment Expander
    with st.expander("📁 Add to Knowledge Base", expanded=False):
        st.caption("Upload files or enter a URL to index.")
        
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            label_visibility="collapsed"
        )
        
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed"
        )

        if st.button("⚡ Index Documents", use_container_width=True):
            os.makedirs("uploads", exist_ok=True)
            pdf_path = None
            txt_path = None

            if uploaded_file:
                path = os.path.join("uploads", uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if uploaded_file.name.lower().endswith(".pdf"):
                    pdf_path = path
                else:
                    txt_path = path

            clean_url = url_input.strip() if url_input and url_input.strip() else None

            if pdf_path or txt_path or clean_url:
                with st.spinner("Processing & indexing documents..."):
                    try:
                        # Calls index_documents from rag.py directly
                        status_msg = index_documents(
                            pdf_path=pdf_path,
                            txt_path=txt_path,
                            url=clean_url
                        )
                        st.success(status_msg)
                        time.sleep(1)
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error indexing: {err}")
            else:
                st.warning("Please choose a file or enter a URL first.")



    # Document list in uploads directory if any
    upload_files = os.listdir("uploads") if os.path.exists("uploads") else []
    if upload_files:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size: 11px; text-transform: uppercase; color: #888888; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px;">
                Uploaded Files
            </div>
        """, unsafe_allow_html=True)
        for uf in upload_files[-4:]:
            st.markdown(f"""
                <div style="font-size: 11px; color: #b0b0b0; padding: 4px 8px; background: #212121; border-radius: 6px; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    📄 {uf}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)

    # Clear Chat option
    if len(st.session_state.messages) > 0:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️  Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==========================================================
# MAIN CHAT AREA
# ==========================================================

# Top Navbar Pill
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; margin-bottom: 8px;">
        <div class="model-pill">
            <span style="color: #10a37f; font-size: 10px;">●</span>
            <span>llama3.1</span>
            <span style="color: #666666;">|</span>
            <span style="color: #9ca3af; font-size: 11px;">RAG Knowledge Base</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# 1. EMPTY STATE / WELCOME HERO (ChatGPT Style)
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div style="text-align: center; margin-top: 3.5rem; margin-bottom: 2.5rem;">
            <div style="width: 58px; height: 58px; margin: 0 auto 16px auto; border-radius: 50%; background: #2f2f2f; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                <span style="font-size: 26px;">✨</span>
            </div>
            <h1 style="font-size: 28px; font-weight: 700; color: #ffffff; margin-bottom: 8px; letter-spacing: -0.5px;">
                What can I help with today?
            </h1>
            <p style="font-size: 14px; color: #9e9e9e; max-width: 500px; margin: 0 auto; line-height: 1.5;">
                Ask questions about your indexed documents, handbook, or custom data.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 4 Quick Suggestion Cards in 2 columns
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡  Summarize key ideas in context\n\nOverview of main topics in knowledge base", use_container_width=True, key="sug_1"):
            st.session_state.pending_prompt = "Summarize the key concepts and topics available in the knowledge base."
            st.rerun()
            
        if st.button("📄  Explain Generative AI\n\nWhat are generative models and architectures?", use_container_width=True, key="sug_2"):
            st.session_state.pending_prompt = "What is Generative AI and how do generative models work according to the context?"
            st.rerun()

    with col2:
        if st.button("🧠  Define Artificial Intelligence\n\nAI definition and fundamentals", use_container_width=True, key="sug_3"):
            st.session_state.pending_prompt = "What is Artificial Intelligence according to the context?"
            st.rerun()

        if st.button("🔍  Examine Knowledge Base Documents\n\nWhat topics are covered in the data?", use_container_width=True, key="sug_4"):
            st.session_state.pending_prompt = "What specific documents and information are stored in this knowledge base?"
            st.rerun()

# 2. RENDER CHAT HISTORY
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(msg["content"])
            # Display source citations if available
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 {len(msg['sources'])} Sources Referenced", expanded=False):
                    for idx, s in enumerate(msg["sources"]):
                        page_str = f" (Page {s['page']})" if s.get("page") else ""
                        st.markdown(f"**Chunk #{idx+1} · `{s['source']}`{page_str}**")
                        st.caption(f'"{s["snippet"]}"')

# ==========================================================
# CHAT INPUT & EXECUTION
# ==========================================================

# Check if a suggestion card was clicked
user_query = st.chat_input("Message ChatGPT RAG...")

if not user_query and st.session_state.pending_prompt:
    user_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_query:
    # 1. Append User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # 2. Generate Assistant Response
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve context sources for citation display
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(user_query)

                sources_list = []
                for d in relevant_docs:
                    src_path = d.metadata.get("source", "Document")
                    page_num = d.metadata.get("page", None)
                    sources_list.append({
                        "source": os.path.basename(src_path) if ("/" in src_path or "\\" in src_path) else src_path,
                        "page": page_num + 1 if page_num is not None else None,
                        "snippet": d.page_content.strip()[:240] + ("..." if len(d.page_content) > 240 else "")
                    })

                # Calls get_answer from rag.py directly (rag.py untouched!)
                raw_answer = get_answer(user_query)

                # Typewriter animation
                st.write_stream(stream_text(raw_answer))

                # Display sources accordion
                if sources_list:
                    with st.expander(f"📚 {len(sources_list)} Sources Referenced", expanded=False):
                        for idx, s in enumerate(sources_list):
                            page_str = f" (Page {s['page']})" if s.get("page") else ""
                            st.markdown(f"**Chunk #{idx+1} · `{s['source']}`{page_str}**")
                            st.caption(f'"{s["snippet"]}"')

                # Store Assistant Message in Session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": raw_answer,
                    "sources": sources_list
                })

            except Exception as e:
                error_text = f"⚠️ **Error generating response**: {str(e)}"
                st.error(error_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                    "sources": []
                })

    st.rerun()

# Disclaimer at bottom like ChatGPT
st.markdown("""
    <div style="text-align: center; color: #737373; font-size: 11px; margin-top: 2rem;">
        ChatGPT RAG can make mistakes. Verify important context from your documents.
    </div>
""", unsafe_allow_html=True)
