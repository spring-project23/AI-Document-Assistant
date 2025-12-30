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

# ================================
# ENVIRONMENT VARIABLES SETUP
# ================================

# For Streamlit Cloud deployment, add these to your secrets.toml file:
# GROQ_API_KEY = "your_groq_key_here"
# LANGCHAIN_TRACING_V2 = "true"
# LANGCHAIN_API_KEY = "lsv2_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# LANGCHAIN_PROJECT = "Document-Assistant"  # Optional: custom project name

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
        return {
            "type": "excel",
            "content": df,
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
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            st.error("No extractable text found in PDF")
            return None
        return {
            "type": "pdf",
            "content": text.strip(),
            "filename": filename,
            "pages": len(reader.pages),
            "text_length": len(text)
        }
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

@traceable(name="Document Q&A Pipeline", run_type="chain")
def ask_question(question: str, language: str, data: dict):
    """Main function — fully traced by LangSmith"""
    data_type = data["type"]
    # Build context
    if data_type == "excel":
        df = data["content"]
        data_summary = f"""
Data Summary:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Column Names: {', '.join(df.columns.tolist())}
Full Data:
{df.to_string(index=False)}
Numeric Summary:
{df.describe().to_string() if not df.select_dtypes(include='number').empty else 'No numeric data'}
"""
    elif data_type == "pdf":
        text = data["content"]
        truncated = text[:8000] + ("..." if len(text) > 8000 else "")
        data_summary = f"""
Document: {data['filename']}
Pages: {data['pages']}
Text Length: {data['text_length']}
Content Preview:
{truncated}
"""
    else:
        st.error("Unknown data type")
        return None
    prompt = f"""You are an accurate AI assistant. Use only the provided document/data to answer.
{data_summary}
Question: {question}
Answer concisely in English. If you cannot answer from the data, say "I cannot determine this from the provided information."
"""
    try:
        # This LLM call is automatically traced by LangSmith
        @traceable(run_type="llm", name=f"Groq - {LLAMA_MODEL_NAME}")
        def call_llm():
            return client.chat.completions.create(
                model=LLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
        response = call_llm()
        english_answer = response.choices[0].message.content.strip()
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
                result["answer_ar"] = english_answer  # fallback
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
            else:
                st.error("Failed to process file.")

# Question interface
if st.session_state.current_data:
    st.header("Ask a Question")
    question = st.text_area("Your question:", height=120, placeholder="e.g., What is the total revenue in 2024?")
    language = st.selectbox("Answer Language", ["en", "ar"], format_func=lambda x: "English" if x == "en" else "Arabic")
    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
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
