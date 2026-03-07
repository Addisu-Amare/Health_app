import os
import asyncio
import tempfile
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from typing import List
import hashlib
import pickle
import langdetect
from langdetect import detect

# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# For Amharic text processing
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2
import io

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
    .amharic-text {
        font-family: 'Noto Sans Ethiopic', 'Abyssinica SIL', 'Nyala', 'Kefa', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 መግለጫ (Description)</h2>", unsafe_allow_html=True)
try:
    st.sidebar.image("utils/ph2.jpg", use_container_width=True)
except:
    st.sidebar.markdown("📷 Image placeholder")
st.sidebar.markdown("""
<p class='sidebar-text' style='font-family: "Noto Sans Ethiopic", sans-serif;'>
የLLM የህክምና ቻትቦት ፈጣን፣ ትክክለኛ እና አስተማማኝ የጤና መረጃ ለመስጠት የተነደፈ AI ረዳት ነው።
</p>
""", unsafe_allow_html=True)

# Ensure async loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
USER_UPLOADS_PATH = "vectorstore/user_uploads"
AMHARIC_DB_PATH = "vectorstore/amharic_medical"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Create directories if they don't exist
os.makedirs(DB_FAISS_PATH, exist_ok=True)
os.makedirs(USER_UPLOADS_PATH, exist_ok=True)
os.makedirs(AMHARIC_DB_PATH, exist_ok=True)

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY የለም። እባክዎ ያስቀምጡት።")
    st.stop()

@st.cache_resource
def load_embedding_model():
    """Load multilingual embedding model that supports Amharic"""
    # Using a multilingual model that supports Amharic
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )

@st.cache_resource
def load_amharic_embedding_model():
    """Load specialized Amharic embedding model"""
    # Alternative: Use a model fine-tuned for Amharic
    return HuggingFaceEmbeddings(
        model_name='xlm-roberta-base'  # Supports Amharic
    )



@st.cache_resource
def load_embedding_model():
    """Load multilingual embedding model that supports Amharic"""
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Then update your load functions
def load_vectorstore():
    embedding_model = load_embedding_model()
    try:
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except:
        return FAISS.from_texts(["initial"], embedding_model)

def load_user_vectorstore():
    embedding_model = load_embedding_model()
    user_db_path = os.path.join(USER_UPLOADS_PATH, "user_docs.faiss")
    try:
        return FAISS.load_local(user_db_path, embedding_model, allow_dangerous_deserialization=True)
    except:
        return FAISS.from_texts(["initial"], embedding_model)

def load_amharic_vectorstore():
    """Load or create Amharic-specific vectorstore"""
    embedding_model = load_embedding_model()  # Use same model
    amharic_db_path = os.path.join(AMHARIC_DB_PATH, "amharic_docs.faiss")
    try:
        return FAISS.load_local(amharic_db_path, embedding_model, allow_dangerous_deserialization=True)
    except:
        return FAISS.from_texts(["initial"], embedding_model)


def save_user_vectorstore(vectorstore, is_amharic=False):
    """Save user uploads vectorstore"""
    if is_amharic:
        save_path = os.path.join(AMHARIC_DB_PATH, "amharic_docs.faiss")
    else:
        save_path = os.path.join(USER_UPLOADS_PATH, "user_docs.faiss")
    vectorstore.save_local(save_path)

def combine_vectorstores(primary_db, secondary_db):
    """
    Combine two vectorstores safely
    
    Args:
        primary_db: The main vectorstore
        secondary_db: The secondary vectorstore to merge into primary
    
    Returns:
        Combined vectorstore
    """
    try:
        # Check if secondary_db exists and has documents (excluding initial placeholder)
        if secondary_db is not None and hasattr(secondary_db, 'index_to_docstore_id'):
            if len(secondary_db.index_to_docstore_id) > 1:  # Has actual documents
                # Merge secondary into primary
                primary_db.merge_from(secondary_db)
                print(f"✅ Merged {len(secondary_db.index_to_docstore_id) - 1} documents from secondary store")
    except Exception as e:
        print(f"⚠️ Warning: Could not merge vectorstores: {str(e)}")
    
    return primary_db



def detect_language(text):
    """Detect if text is Amharic or English"""
    try:
        if len(text.strip()) < 10:  # Too short for reliable detection
            return "unknown"
        
        # Check for Amharic Unicode range
        amharic_range = range(0x1200, 0x137F)  # Ethiopic block
        amharic_chars = sum(1 for char in text if ord(char) in amharic_range)
        
        if amharic_chars > len(text) * 0.1:  # At least 10% Amharic characters
            return "amharic"
        
        # Use langdetect as fallback
        lang = detect(text)
        if lang == 'am':
            return "amharic"
        return lang
    except:
        return "unknown"

def extract_text_from_pdf_amharic(pdf_path):
    """Extract text from PDF with Amharic support"""
    text = ""
    
    # Try PyPDF2 first (works for text-based PDFs)
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {str(e)}")
    
    # If text extraction yielded little content, try OCR
    if len(text.strip()) < 100:
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            for image in images:
                # Use Tesseract with Amharic language support
                try:
                    page_text = pytesseract.image_to_string(image, lang='amh+eng')
                    text += page_text + "\n"
                except Exception as e:
                    st.warning(f"OCR failed: {str(e)}")
                    # Fallback to English-only OCR
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"OCR processing failed: {str(e)}")
    
    return text

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text (supports Amharic)"""
    documents = []
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if uploaded_file.type == "application/pdf":
            # Use enhanced PDF extraction for Amharic
            text = extract_text_from_pdf_amharic(tmp_file_path)
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name, "user_uploaded": True}
                )
                documents = [doc]
        elif uploaded_file.type == "text/plain":
            # Read text file with UTF-8 encoding (important for Amharic)
            content = uploaded_file.getvalue().decode('utf-8')
            doc = Document(
                page_content=content,
                metadata={"source": uploaded_file.name, "user_uploaded": True}
            )
            documents = [doc]
        else:
            st.error(f"ያልተደገፈ የፋይል አይነት: {uploaded_file.type}")
            return []
        
        # Detect language
        if documents:
            language = detect_language(documents[0].page_content[:1000])
            documents[0].metadata["language"] = language
        
        return documents
    except Exception as e:
        st.error(f"ፋይል በማስኬድ ላይ ስህተት: {str(e)}")
        return []
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)

def chunk_documents_amharic(documents: List[Document]):
    """Split documents into chunks with Amharic-aware splitting"""
    # Custom separators for Amharic text
    amharic_separators = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        "።",     # Amharic period
        "፣",     # Amharic comma
        "፤",     # Amharic semicolon
        "፦",     # Amharic colon
        " ",     # Space
        ""       # Character level
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for better Amharic processing
        chunk_overlap=100,
        length_function=len,
        separators=amharic_separators
    )
    return text_splitter.split_documents(documents)

def add_documents_to_vectorstore(documents, vectorstore, is_amharic=False):
    """Add documents to appropriate vectorstore based on language"""
    if documents and vectorstore:
        # Use language-appropriate chunking
        chunks = chunk_documents_amharic(documents) if is_amharic else chunk_documents_amharic(documents)
        vectorstore.add_documents(chunks)
        return True
    return False

def get_prompt_template(language="english"):
    """Get prompt template in appropriate language"""
    if language == "amharic":
        return PromptTemplate(
            template="""እርስዎ የህክምና AI ረዳት ነዎት። የተሰጠውን መረጃ በመጠቀም የተጠቃሚውን ጥያቄ ይመልሱ።
መልሱን ካላወቁ "አላውቀውም" ይበሉ እንጂ የራስዎን መልስ አይፍጠሩ። ሁልጊዜ በተሰጠው መረጃ ውስጥ ይቆዩ።

**የተሰጠ መረጃ (ከህክምና የእውቀት መረብ እና ከተጠቃሚ ከተላኩ ሰነዶች):**
{context}

**ጥያቄ:**
{question}

እባክዎ **አጭር እና ጠቃሚ መልስ** ይስጡ።
            """,
            input_variables=["context", "question"]
        )
    else:
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
        temperature=0.3,  # Lower temperature for more accurate medical responses
        model_name="llama-3.3-70b-versatile"  # This model supports multiple languages
    )

def format_sources(source_documents):
    """Format sources with language indication"""
    if not source_documents:
        return "**ምንጮች:** ምንም ምንጭ አልተገኘም።" if st.session_state.get('current_language') == 'amharic' else "**Sources:** No sources found."
    
    formatted_sources = "\n\n**ምንጮች:**" if st.session_state.get('current_language') == 'amharic' else "\n\n**Sources:**"
    user_sources = []
    main_sources = []
    amharic_sources = []
    
    for idx, doc in enumerate(source_documents, start=1):
        source_name = doc.metadata.get('source', 'Unknown Source')
        language = doc.metadata.get('language', 'unknown')
        
        if doc.metadata.get('user_uploaded', False):
            if language == 'amharic':
                amharic_sources.append(f"📁 **የተጠቃሚ አማርኛ ሰነድ {idx}:** {source_name}")
            else:
                user_sources.append(f"📁 **የተጠቃሚ ሰነድ {idx}:** {source_name}")
        else:
            main_sources.append(f"📚 **ምንጭ {idx}:** {source_name}")
    
    if main_sources:
        formatted_sources += "\n" + "\n".join(main_sources)
    if user_sources:
        formatted_sources += "\n" + "\n".join(user_sources)
    if amharic_sources:
        formatted_sources += "\n" + "\n".join(amharic_sources)
    
    return formatted_sources

def main():
    st.title("💬 ሜዲቦት - የAI ጤና ረዳት (AI Health Assistant)")
    
    # Language selector
    col1, col2 = st.columns([3, 1])
    with col2:
        language = st.selectbox("ቋንቋ / Language", ["English", "አማርኛ"])
        st.session_state['current_language'] = 'amharic' if language == "አማርኛ" else 'english'
    
    if language == "አማርኛ":
        st.markdown("""
            **ማንኛውንም ከህክምና ጋር የተያያዘ ጥያቄ ይጠይቁ፣ እኔም በአስተማማኝ መረጃ ላይ በመመስረት መልስ እሰጣለሁ!**
            🤖🩺 *በ AI እና Meta(llama) የተጎላበተ*
        """)
    else:
        st.markdown("""
            **Ask any medical-related question, and I'll provide insights based on reliable information!**
            🤖🩺 *Powered by AI & Meta(llama)*
        """)

    # Sidebar for document upload
    with st.sidebar:
        if language == "አማርኛ":
            st.markdown("### 📤 የህክምና ሰነዶችን ይላኩ")
            st.markdown("የራስዎን የህክምና ሰነዶች በማከል የእውቀት መረቡን ያሳድጉ፡")
            upload_text = "PDF ወይም TXT ፋይሎችን ይምረጡ"
            support_text = "የህክምና ሰነዶችን ይላኩ (PDF ወይም TXT) ግላዊ መልስ ለማግኘት"
            process_button = "📥 የተላኩ ሰነዶችን አስኬድ"
        else:
            st.markdown("### 📤 Upload Medical Documents")
            st.markdown("Add your own medical documents to enhance the knowledge base:")
            upload_text = "Choose PDF or TXT files"
            support_text = "Upload medical documents (PDF or TXT) to get personalized responses"
            process_button = "📥 Process Uploaded Documents"
        
        uploaded_files = st.file_uploader(
            upload_text,
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help=support_text
        )
        
        if uploaded_files:
            if st.button(process_button, type="primary"):
                with st.spinner("ሰነዶችን በማስኬድ ላይ..." if language == "አማርኛ" else "Processing documents..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load appropriate vectorstore based on language detection
                    user_db = load_user_vectorstore()
                    amharic_db = load_amharic_vectorstore()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Process file
                        documents = process_uploaded_file(uploaded_file)
                        
                        if documents:
                            # Detect language and add to appropriate store
                            language_detected = documents[0].metadata.get('language', 'unknown')
                            
                            if language_detected == 'amharic':
                                if add_documents_to_vectorstore(documents, amharic_db, is_amharic=True):
                                    st.success(f"✅ ተጨምሯል: {uploaded_file.name} (አማርኛ)")
                                    save_user_vectorstore(amharic_db, is_amharic=True)
                            else:
                                if add_documents_to_vectorstore(documents, user_db):
                                    st.success(f"✅ Added: {uploaded_file.name}")
                                    save_user_vectorstore(user_db)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("ሂደቱ ተጠናቋል!" if language == "አማርኛ" else "Processing complete!")
                    st.success("🎉 ሁሉም ሰነዶች ተሰርተው በእውቀት መረቡ ውስጥ ተጨምረዋል!" if language == "አማርኛ" else "🎉 All documents processed and added to knowledge base!")
                    
                    # Clear the uploader
                    st.rerun()
        
        st.markdown("---")
        
        if language == "አማርኛ":
            st.markdown("### 🔍 ስለ ሜዲቦት:")
            st.markdown("""
            - **Meta-llama (Groq)** በመጠቀም የህክምና ጥያቄዎችን ይመልሳል
            - ከህክምና የእውቀት መረብ ውስጥ መረጃ ያመጣል
            - **በተጠቃሚ የተላኩ ሰነዶችን** ይደግፋል (PDF, TXT)
            - **አማርኛ ሰነዶችን** ማስኬድ ይችላል
            - **ፈጣን፣ አስተማማኝ እና ትክክለኛ** መልስ ይሰጣል
            """)
        else:
            st.markdown("### 🔍 About Medibot:")
            st.markdown("""
            - Uses **Meta-llama (Groq)** to answer medical queries
            - Retrieves relevant medical data from knowledge base
            - Supports **user-uploaded documents** (PDF, TXT)
            - Can process **Amharic medical documents**
            - Provides **fast, reliable, and contextual responses**
            """)
        
        # Show uploaded documents info
        user_db = load_user_vectorstore()
        amharic_db = load_amharic_vectorstore()
        
        if language == "አማርኛ":
            if len(amharic_db.index_to_docstore_id) > 1:
                st.markdown("---")
                st.markdown("### 📚 የተላኩ አማርኛ ሰነዶች")
                st.info(f"✅ {len(amharic_db.index_to_docstore_id) - 1} የአማርኛ ሰነድ ክፍሎች ተጨምረዋል")
            if len(user_db.index_to_docstore_id) > 1:
                st.markdown("---")
                st.markdown("### 📚 የተላኩ እንግሊዝኛ ሰነዶች")
                st.info(f"✅ {len(user_db.index_to_docstore_id) - 1} የእንግሊዝኛ ሰነድ ክፍሎች ተጨምረዋል")
        else:
            if len(amharic_db.index_to_docstore_id) > 1:
                st.markdown("---")
                st.markdown("### 📚 Uploaded Amharic Documents")
                st.info(f"✅ {len(amharic_db.index_to_docstore_id) - 1} Amharic document chunks available")
            if len(user_db.index_to_docstore_id) > 1:
                st.markdown("---")
                st.markdown("### 📚 Uploaded English Documents")
                st.info(f"✅ {len(user_db.index_to_docstore_id) - 1} English document chunks available")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_query = st.chat_input("የህክምና ጥያቄዎን ይጻፉ..." if language == "አማርኛ" else "Type your medical query...")

    if user_query:
        st.chat_message("user").markdown(f"**{'እርስዎ' if language == 'አማርኛ' else 'You'}:** {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("🤖 ሜዲቦት እያሰበ ነው..." if language == "አማርኛ" else "🤖 Medibot is thinking..."):
            try:
                # Detect query language
                query_lang = detect_language(user_query)
                
                # Load appropriate vectorstores
                main_db = load_vectorstore()
                user_db = load_user_vectorstore()
                amharic_db = load_amharic_vectorstore()
                
                # Combine based on query language
                if query_lang == 'amharic':
                    combined_db = combine_vectorstores(amharic_db, user_db)
                else:
                    combined_db = combine_vectorstores(main_db, user_db)
                    # Also include Amharic docs if needed
                    if len(amharic_db.index_to_docstore_id) > 1:
                        combined_db.merge_from(amharic_db)
                
                if combined_db is None:
                    st.error("❌ ስህተት: የቬክተር ማከማቻ መጫን አልተቻለም።" if language == "አማርኛ" else "❌ Error: Vector store failed to load.")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=combined_db.as_retriever(search_kwargs={'k': 5}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': get_prompt_template(query_lang if query_lang in ['amharic', 'english'] else 'english')}
                )

                response = qa_chain.invoke({'query': user_query})
                result = response.get("result", "⚠️ ምንም መልስ አልተገኘም።" if language == "አማርኛ" else "⚠️ No response generated.")
                sources = response.get("source_documents", [])

                formatted_response = f"**ሜዲቦት:** {result}\n\n{format_sources(sources)}" if language == "አማርኛ" else f"**Medibot:** {result}\n\n{format_sources(sources)}"
                st.chat_message("assistant").markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            except Exception as e:
                st.error(f"⚠️ ስህተት: {str(e)}" if language == "አማርኛ" else f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()




