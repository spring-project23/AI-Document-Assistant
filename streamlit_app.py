import os
import io
import pandas as pd
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from gtts import gTTS
from groq import Groq
import tempfile
import streamlit as st
# LangSmith integration
from langsmith import traceable
# LangChain imports for RAG
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# ================================
# ENVIRONMENT VARIABLES SETUP
# ================================
# For Streamlit Cloud deployment, add these to your secrets.toml file:
# GROQ_API_KEY = "your_groq_key_here"
# LANGCHAIN_TRACING_V2 = "true"
# LANGCHAIN_API_KEY = "lsv2_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# LANGCHAIN_PROJECT = "Document-Assistant" # Optional: custom project name
# Groq API Key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your Streamlit secrets.")
    st.stop()
# LangSmith setup - Set environment variables from secrets for deployment
# This ensures tracing works on Streamlit Cloud without relying on .env
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "Document-Assistant")
# Only enable tracing if API key is provided
if not os.environ["LANGCHAIN_API_KEY"]:
    st.warning("LANGCHAIN_API_KEY not found in secrets. LangSmith tracing will be disabled.")
# Groq Client
LLAMA_MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)
# Session state
if "current_data" not in st.session_state:
    st.session_state.current_data = None
def build_vectorstore(data: dict):
    """Build FAISS vectorstore from documents using LangChain RAG components."""
    documents = data["documents"]
    if not documents:
        raise ValueError("No documents to index.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore
def process_excel(file_content, filename):
    if not (filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls')):
        st.error("Only .xlsx or .xls files allowed")
        return None
    try:
        df_raw = pd.read_excel(io.BytesIO(file_content), header=None, dtype=str)
        if df_raw.empty:
            st.error("Uploaded file is empty")
            return None
        # Smart header detection
        header_row_index = None
        for i in range(min(5, len(df_raw))):
            row = df_raw.iloc[i]
            non_nan_count = row.notna().sum()
            if 2 <= non_nan_count <= 20:
                avg_len = row.dropna().astype(str).str.len().mean()
                if avg_len < 30:
                    header_row_index = i
                    break
        if header_row_index is not None:
            headers = df_raw.iloc[header_row_index].fillna("").tolist()
            headers = [str(h).strip() if str(h).strip() else f"Column_{j}" for j, h in enumerate(headers)]
            df = df_raw.iloc[header_row_index + 1:].copy()
            df.columns = headers[:len(df.columns)]
        else:
            df = df_raw.copy()
            df.columns = [f"Column_{j+1}" for j in range(df.shape[1])]
        df = df.dropna(how='all').reset_index(drop=True)
        df = df.loc[:, df.columns.notna()]
        df = df.loc[:, (df != "").any(axis=0)]
        if df.empty:
            st.error("No valid data found after cleaning")
            return None
        # Convert to markdown for RAG
        markdown_content = df.to_markdown(index=False)
        doc = Document(
            page_content=markdown_content,
            metadata={
                "type": "excel",
                "filename": filename,
                "rows": len(df),
                "columns": list(df.columns)
            }
        )
        return {
            "type": "excel",
            "documents": [doc],
            "filename": filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        st.error(f"Error processing Excel: {str(e)}")
        return None
def process_pdf(file_content, filename):
    if not filename.lower().endswith('.pdf'):
        st.error("Only .pdf files allowed")
        return None
    try:
        reader = PdfReader(io.BytesIO(file_content))
        if len(reader.pages) == 0:
            st.error("PDF is empty")
            return None
        documents = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                doc = Document(
                    page_content=page_text.strip(),
                    metadata={
                        "type": "pdf",
                        "filename": filename,
                        "page": i + 1
                    }
                )
                documents.append(doc)
        if not documents:
            st.error("No extractable text found in PDF")
            return None
        return {
            "type": "pdf",
            "documents": documents,
            "filename": filename,
            "pages": len(reader.pages),
            "text_length": sum(len(doc.page_content) for doc in documents)
        }
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
@traceable(name="Document Q&A Pipeline", run_type="chain")
def ask_question(question: str, language: str, data: dict):
    """Main function — fully traced by LangSmith. Uses LangChain RAG for retrieval."""
    data_type = data["type"]
    try:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLAMA_MODEL_NAME,
            temperature=0.7,
            max_tokens=1024
        )
        prompt = PromptTemplate(
            template="""You are an accurate AI assistant. Use only the provided context from the document/data to answer the question.

Context: {context}

Question: {question}

Answer concisely in English. If you cannot answer from the context, say "I cannot determine this from the provided information.""",
            input_variables=["context", "question"],
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        @traceable(run_type="chain", name="RAG Chain")
        def run_qa():
            return qa_chain.invoke({"query": question})
        
        result_dict = run_qa()
        english_answer = result_dict["result"].strip()
        result = {
            "question": question,
            "answer_en": english_answer,
            "data_type": data_type
        }
        # Translation to Arabic
        if language == "ar":
            try:
                arabic_answer = GoogleTranslator(source='en', target='ar').translate(english_answer)
                result["answer_ar"] = arabic_answer
            except Exception as e:
                st.warning(f"Translation failed: {e}")
                result["answer_ar"] = english_answer # fallback
        # Text-to-Speech
        text_to_speak = result.get("answer_ar") if language == "ar" else english_answer
        tts_lang = "ar" if language == "ar" else "en"
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)
        try:
            tts = gTTS(text=text_to_speak, lang=tts_lang, slow=False)
            tts.save(temp_path)
            with open(temp_path, "rb") as f:
                result["audio_bytes"] = f.read()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return result
    except Exception as e:
        st.error(f"Failed to generate answer: {str(e)}")
        return None
# ================================
# Streamlit UI
# ================================
st.title("AI Document Assistant")
st.caption("Upload an Excel or PDF and ask questions in English or Arabic")
uploaded_file = st.file_uploader(
    "Upload Excel (.xlsx, .xls) or PDF",
    type=["xlsx", "xls", "pdf"]
)
if uploaded_file is not None:
    file_content = uploaded_file.read()
    filename = uploaded_file.name
    if st.button("Process File", type="primary"):
        with st.spinner("Processing file..."):
            if filename.lower().endswith(('.xlsx', '.xls')):
                processed = process_excel(file_content, filename)
            else:
                processed = process_pdf(file_content, filename)
            if processed:
                st.session_state.current_data = processed
                st.success(f"Processed: {filename}")
                if processed["type"] == "excel":
                    st.info(f"Rows: {processed['rows']} | Columns: {', '.join(processed['columns'][:8])}{'...' if len(processed['columns']) > 8 else ''}")
                else:
                    st.info(f"Pages: {processed['pages']} | Characters: {processed['text_length']:,}")
                
                # Build vectorstore for RAG
                with st.spinner("Building retrieval index... (This may take a moment)"):
                    try:
                        st.session_state.vectorstore = build_vectorstore(processed)
                        st.success("Retrieval index built successfully!")
                    except Exception as e:
                        st.error(f"Failed to build index: {str(e)}")
                        st.session_state.current_data = None
                        st.session_state.pop("vectorstore", None)
            else:
                st.error("Failed to process file.")
# Question interface
if st.session_state.current_data and "vectorstore" in st.session_state:
    st.header("Ask a Question")
    question = st.text_area("Your question:", height=120, placeholder="e.g., What is the total revenue in 2024?")
    language = st.selectbox("Answer Language", ["en", "ar"], format_func=lambda x: "English" if x == "en" else "Arabic")
    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and thinking..."):
                result = ask_question(question.strip(), language, st.session_state.current_data)
                if result:
                    answer = result.get("answer_ar") if language == "ar" else result["answer_en"]
                    st.markdown(f"**Answer ({'Arabic' if language == 'ar' else 'English'}):**")
                    st.markdown(answer)
                    st.audio(result["audio_bytes"], format="audio/mp3")
else:
    st.info("Upload and process a file to begin asking questions.")
# Optional: Show LangSmith link if tracing is enabled
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    st.sidebar.caption(f"LangSmith tracing enabled → [View traces](https://smith.langchain.com/projects/p/{project_name})")
