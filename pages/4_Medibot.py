import os
import asyncio
import tempfile
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import List
import hashlib
import pickle

# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# Load environment variables
load_dotenv(find_dotenv())
nest_asyncio.apply()

# Custom CSS for better UI
st.markdown("""
<style>
    .upload-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)
st.sidebar.image("utils/ph2.jpg", use_container_width=True)
st.sidebar.markdown("<p class='sidebar-text'>The LLM Medical Chatbot is an AI-powered assistant designed to provide instant, accurate, and reliable healthcare insights.</p>", unsafe_allow_html=True)

# Ensure async loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
USER_UPLOADS_PATH = "vectorstore/user_uploads"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Create directories if they don't exist
os.makedirs(DB_FAISS_PATH, exist_ok=True)
os.makedirs(USER_UPLOADS_PATH, exist_ok=True)

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing. Please set it in your environment.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    """Load the embedding model"""
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_vectorstore():
    """Load the main vectorstore"""
    embedding_model = load_embedding_model()
    try:
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except:
        # Return empty vectorstore if none exists
        return FAISS.from_texts(["initial"], embedding_model)

def load_user_vectorstore():
    """Load or create user uploads vectorstore"""
    embedding_model = load_embedding_model()
    user_db_path = os.path.join(USER_UPLOADS_PATH, "user_docs.faiss")
    try:
        return FAISS.load_local(user_db_path, embedding_model, allow_dangerous_deserialization=True)
    except:
        return FAISS.from_texts(["initial"], embedding_model)

def save_user_vectorstore(vectorstore):
    """Save user uploads vectorstore"""
    user_db_path = os.path.join(USER_UPLOADS_PATH, "user_docs.faiss")
    vectorstore.save_local(user_db_path)

def combine_vectorstores(main_db, user_db):
    """Combine main and user vectorstores"""
    if user_db is not None and len(user_db.index_to_docstore_id) > 1:  # >1 to exclude initial placeholder
        # Merge user docs with main db
        main_db.merge_from(user_db)
    return main_db

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text"""
    documents = []
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_file_path)
            documents = loader.load()
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return []
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["user_uploaded"] = True
            doc.metadata["file_hash"] = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        return documents
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)

def chunk_documents(documents: List[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def add_documents_to_vectorstore(documents, vectorstore):
    """Add documents to vectorstore"""
    if documents and vectorstore:
        chunks = chunk_documents(documents)
        vectorstore.add_documents(chunks)
        return True
    return False

def get_prompt_template():
    return PromptTemplate(
        template="""You are a medical AI assistant. Use the provided context to answer the user's question.
If you don't know the answer, say "I don't know" instead of making one up. Always stay within the given context.

**Context (from medical knowledge base and user-uploaded documents):**
{context}

**Question:**
{question}

Please provide a **concise and informative response** based only on the provided context.
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    return ChatGroq(
        temperature=0.5,
        model_name="llama-3.3-70b-versatile"
    )

def format_sources(source_documents):
    if not source_documents:
        return "**Sources:** No sources found."
    
    formatted_sources = "\n\n**Sources:**"
    user_sources = []
    main_sources = []
    
    for idx, doc in enumerate(source_documents, start=1):
        source_name = doc.metadata.get('source', 'Unknown Source')
        if doc.metadata.get('user_uploaded', False):
            user_sources.append(f"📁 **User Upload {idx}:** {source_name}")
        else:
            main_sources.append(f"📚 **Source {idx}:** {source_name}")
    
    if main_sources:
        formatted_sources += "\n" + "\n".join(main_sources)
    if user_sources:
        formatted_sources += "\n" + "\n".join(user_sources)
    
    return formatted_sources

def main():
    st.title("💬 Medibot - AI Health Assistant")
    st.markdown("""
        **Ask any medical-related question, and I'll provide insights based on reliable information!**
        🤖🩺 *Powered by AI & Meta(llama)*
    """)

    # Sidebar for document upload
    with st.sidebar:
        st.markdown("### 📤 Upload Medical Documents")
        st.markdown("Add your own medical documents to enhance the knowledge base:")
        
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload medical documents (PDF or TXT) to get personalized responses"
        )
        
        if uploaded_files:
            if st.button("📥 Process Uploaded Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load user vectorstore
                    user_db = load_user_vectorstore()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Process file
                        documents = process_uploaded_file(uploaded_file)
                        
                        if documents:
                            # Add to vectorstore
                            if add_documents_to_vectorstore(documents, user_db):
                                st.success(f"✅ Added: {uploaded_file.name}")
                            else:
                                st.error(f"❌ Failed to add: {uploaded_file.name}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Save user vectorstore
                    save_user_vectorstore(user_db)
                    
                    status_text.text("Processing complete!")
                    st.success("🎉 All documents processed and added to knowledge base!")
                    
                    # Clear the uploader
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔍 About Medibot:")
        st.markdown("""
        - Uses **Meta-llama (Groq)** to answer medical queries
        - Retrieves relevant medical data from knowledge base
        - Supports **user-uploaded documents** (PDF, TXT)
        - Provides **fast, reliable, and contextual responses**
        """)
        
        # Show uploaded documents info
        user_db = load_user_vectorstore()
        if len(user_db.index_to_docstore_id) > 1:
            st.markdown("---")
            st.markdown("### 📚 Your Uploaded Documents")
            st.info(f"✅ {len(user_db.index_to_docstore_id) - 1} document chunks from your uploads are available")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_query = st.chat_input("Type your medical query...")

    if user_query:
        st.chat_message("user").markdown(f"**You:** {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("🤖 Medibot is thinking..."):
            try:
                # Load both vectorstores
                main_db = load_vectorstore()
                user_db = load_user_vectorstore()
                
                # Combine them
                combined_db = combine_vectorstores(main_db, user_db)
                
                if combined_db is None:
                    st.error("❌ Error: Vector store failed to load.")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=combined_db.as_retriever(search_kwargs={'k': 5}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': get_prompt_template()}
                )

                response = qa_chain.invoke({'query': user_query})
                result = response.get("result", "⚠️ No response generated.")
                sources = response.get("source_documents", [])

                formatted_response = f"**Medibot:** {result}\n\n{format_sources(sources)}"
                st.chat_message("assistant").markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()
