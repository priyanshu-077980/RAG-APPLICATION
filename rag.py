from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


# 1. Embeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 2. ChromaDB

vectorstore = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# 3. LLM

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)


# 4. Prompt

prompt = ChatPromptTemplate.from_template("""
Answer the question using the given context.

Context:
{context}

Question:
{question}

If answer is not available in context,
say you don't know.
""")


# 5. Document Loading Function

def load_documents(pdf_path=None, txt_path=None, url=None):

    documents = []

    # PDF Loader
    if pdf_path:

        loader = PyPDFLoader(pdf_path)

        documents.extend(loader.load())


    # Text Loader
    if txt_path:

        loader = TextLoader(
            txt_path,
            encoding="utf-8"
        )

        documents.extend(loader.load())


    # URL Loader
    if url:

        loader = WebBaseLoader(url)

        documents.extend(loader.load())


    return documents


# 6. Indexing Function

def index_documents(pdf_path=None, txt_path=None, url=None):

    documents = load_documents(
        pdf_path,
        txt_path,
        url
    )

    if not documents:
        return "No documents found!"

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # Store in ChromaDB
    vectorstore.add_documents(chunks)

    return f"{len(chunks)} chunks stored successfully!"


# 7. RAG Function

def get_answer(question):

    # Retrieve Documents
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    # Extract Context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Create Prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # Generate Answer
    response = llm.invoke(messages)

    return response.content