import os
import io
import requests
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq
import tempfile

# Load environment variables
load_dotenv()

# Groq Configuration
GROQ_API_KEY = (
    st.secrets.get("GROQ_API_KEY")
    or os.getenv("GROQ_API_KEY")
)
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

LLAMA_MODEL_NAME = "llama-3.3-70b-versatile"  # Llama 3.3 model on Groq
client = Groq(api_key=GROQ_API_KEY)

# Initialize session state
if "current_data" not in st.session_state:
    st.session_state.current_data = None

def process_excel(file_content, filename):
    """Process Excel file similar to the original endpoint."""
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
        if df.empty or df.shape[0] == 0:
            st.error("No valid data found")
            return None
        return {
            "type": "excel",
            "content": df,
            "filename": filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def process_pdf(file_content, filename):
    """Process PDF file similar to the original endpoint."""
    if not filename.lower().endswith('.pdf'):
        st.error("Only .pdf files allowed")
        return None
    try:
        reader = PdfReader(io.BytesIO(file_content))
        if len(reader.pages) == 0:
            st.error("Uploaded file is empty")
            return None
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        if not text.strip():
            st.error("No text found in PDF")
            return None
        return {
            "type": "pdf",
            "content": text,
            "filename": filename,
            "pages": len(reader.pages),
            "text_length": len(text)
        }
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def ask_question(question, language, data):
    """Generate answer using Groq, translate if needed, and create TTS."""
    data_type = data["type"]
    # Build data context
    if data_type == "excel":
        df = data["content"]
        data_summary = f"""
Data Summary:
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Column Names: {', '.join(df.columns.tolist())}
- Column Types: {df.dtypes.to_dict()}
All rows:
{df.to_string()}
Statistical Summary (if numeric):
{df.describe().to_string() if not df.select_dtypes(include='number').empty else 'No numeric columns.'}
"""
    elif data_type == "pdf":
        text = data["content"]
        # Truncate if too long to avoid token limits
        text_truncated = text[:8000] + "..." if len(text) > 8000 else text
        data_summary = f"""
Document Summary:
- Filename: {data.get('filename', 'Unknown')}
- Number of Pages: {data.get('pages', 0)}
- Text Length: {data.get('text_length', 0)}
Content:
{text_truncated}
"""
    else:
        st.error("Unsupported data type")
        return None
    prompt = f"""You are a smart assistant. Answer the user's question accurately using the document or data below.
{data_summary}
User Question: {question}
Answer clearly and concisely in English. If the question is unanswerable, say so."""
    try:
        # Call Groq API
        chat_completion = client.chat.completions.create(
            model=LLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        english_answer = chat_completion.choices[0].message.content.strip()
        result = {
            "question": question,
            "answer_en": english_answer,
            "data_type": data_type
        }
        # Translate to Arabic if requested
        arabic_answer = None
        if language == "ar":
            try:
                arabic_answer = GoogleTranslator(source='en', target='ar').translate(english_answer)
                result["answer_ar"] = arabic_answer
            except Exception as e:
                st.warning(f"Translation failed: {str(e)}")
        # Generate TTS using temporary file
        text_for_tts = arabic_answer if language == "ar" and arabic_answer else english_answer
        tts_lang = "ar" if language == "ar" else "en"
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)
        try:
            tts = gTTS(text=text_for_tts, lang=tts_lang)
            tts.save(temp_path)
            with open(temp_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            result["audio_bytes"] = audio_bytes
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return result
    except Exception as e:
        if "groq" in str(e).lower() or "api" in str(e).lower():
            st.error(f"Groq API error: {str(e)}")
        else:
            st.error(f"Error generating response: {str(e)}")
        return None

# Streamlit UI
st.title("AI Document Assistant")

# File Upload
uploaded_file = st.file_uploader(
    "Choose an Excel or PDF file",
    type=['xlsx', 'xls', 'pdf'],
    help="Upload a file to start asking questions."
)

if uploaded_file is not None:
    file_content = uploaded_file.read()
    filename = uploaded_file.name
    if st.button("Process File"):
        if filename.lower().endswith(('.xlsx', '.xls')):
            processed_data = process_excel(file_content, filename)
        else:
            processed_data = process_pdf(file_content, filename)
        if processed_data:
            st.session_state.current_data = processed_data
            st.success(f"File processed: {filename}")
            if processed_data["type"] == "excel":
                st.info(f"Rows: {processed_data['rows']}, Columns: {', '.join(processed_data['columns'])}")
            else:
                st.info(f"Pages: {processed_data['pages']}, Text Length: {processed_data['text_length']}")

# Question Asking
if st.session_state.current_data is not None:
    st.header("Ask a Question")
    question = st.text_area("Your question:", height=100)
    language = st.selectbox("Language", ["en", "ar"], format_func=lambda x: "English" if x == "en" else "Arabic")
    if st.button("Ask"):
        if question.strip():
            with st.spinner("Generating answer..."):
                result = ask_question(question, language, st.session_state.current_data)
                if result:
                    if language == "ar" and "answer_ar" in result:
                        st.write("**Answer (Arabic):**")
                        st.write(result["answer_ar"])
                    else:
                        st.write("**Answer (English):**")
                        st.write(result["answer_en"])
                    st.audio(result["audio_bytes"], format="audio/mp3")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload and process a file first.")
# import os
# import io
# import requests
# import streamlit as st
# import pandas as pd
# from PyPDF2 import PdfReader
# from deep_translator import GoogleTranslator
# from gtts import gTTS
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Ollama Configuration
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
# OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
# OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
# QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3-vl:235b-cloud")

# # Initialize session state
# if "current_data" not in st.session_state:
#     st.session_state.current_data = None

# def process_excel(file_content, filename):
#     """Process Excel file similar to the original endpoint."""
#     if not (filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls')):
#         st.error("Only .xlsx or .xls files allowed")
#         return None

#     try:
#         df_raw = pd.read_excel(io.BytesIO(file_content), header=None, dtype=str)
#         if df_raw.empty:
#             st.error("Uploaded file is empty")
#             return None

#         # Smart header detection
#         header_row_index = None
#         for i in range(min(5, len(df_raw))):
#             row = df_raw.iloc[i]
#             non_nan_count = row.notna().sum()
#             if 2 <= non_nan_count <= 20:
#                 avg_len = row.dropna().astype(str).str.len().mean()
#                 if avg_len < 30:
#                     header_row_index = i
#                     break

#         if header_row_index is not None:
#             headers = df_raw.iloc[header_row_index].fillna("").tolist()
#             headers = [str(h).strip() if str(h).strip() else f"Column_{j}" for j, h in enumerate(headers)]
#             df = df_raw.iloc[header_row_index + 1:].copy()
#             df.columns = headers[:len(df.columns)]
#         else:
#             df = df_raw.copy()
#             df.columns = [f"Column_{j+1}" for j in range(df.shape[1])]

#         df = df.dropna(how='all').reset_index(drop=True)
#         df = df.loc[:, df.columns.notna()]
#         df = df.loc[:, (df != "").any(axis=0)]

#         if df.empty or df.shape[0] == 0:
#             st.error("No valid data found")
#             return None

#         return {
#             "type": "excel",
#             "content": df,
#             "filename": filename,
#             "rows": len(df),
#             "columns": list(df.columns)
#         }
#     except Exception as e:
#         st.error(f"Error processing file: {str(e)}")
#         return None

# def process_pdf(file_content, filename):
#     """Process PDF file similar to the original endpoint."""
#     if not filename.lower().endswith('.pdf'):
#         st.error("Only .pdf files allowed")
#         return None

#     try:
#         reader = PdfReader(io.BytesIO(file_content))
#         if len(reader.pages) == 0:
#             st.error("Uploaded file is empty")
#             return None

#         # Extract text from all pages
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() + "\n"

#         if not text.strip():
#             st.error("No text found in PDF")
#             return None

#         return {
#             "type": "pdf",
#             "content": text,
#             "filename": filename,
#             "pages": len(reader.pages),
#             "text_length": len(text)
#         }
#     except Exception as e:
#         st.error(f"Error processing file: {str(e)}")
#         return None

# def ask_question(question, language, data):
#     """Generate answer using Ollama, translate if needed, and create TTS."""
#     data_type = data["type"]

#     # Build data context
#     if data_type == "excel":
#         df = data["content"]
#         data_summary = f"""
# Data Summary:
# - Total Rows: {df.shape[0]}
# - Total Columns: {df.shape[1]}
# - Column Names: {', '.join(df.columns.tolist())}
# - Column Types: {df.dtypes.to_dict()}
# All rows:
# {df.to_string()}
# Statistical Summary (if numeric):
# {df.describe().to_string() if not df.select_dtypes(include='number').empty else 'No numeric columns.'}
# """
#     elif data_type == "pdf":
#         text = data["content"]
#         # Truncate if too long to avoid token limits
#         text_truncated = text[:8000] + "..." if len(text) > 8000 else text
#         data_summary = f"""
# Document Summary:
# - Filename: {data.get('filename', 'Unknown')}
# - Number of Pages: {data.get('pages', 0)}
# - Text Length: {data.get('text_length', 0)}
# Content:
# {text_truncated}
# """
#     else:
#         st.error("Unsupported data type")
#         return None

#     prompt = f"""You are a smart assistant. Answer the user's question accurately using the document or data below.
# {data_summary}
# User Question: {question}
# Answer clearly and concisely in English. If the question is unanswerable, say so."""

#     try:
#         # Call Ollama API
#         ollama_payload = {
#             "model": QWEN_MODEL_NAME,
#             "prompt": prompt,
#             "stream": False
#         }
#         response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=ollama_payload, timeout=60)
#         response.raise_for_status()
#         ollama_result = response.json()
#         english_answer = ollama_result.get("response", "").strip()

#         result = {
#             "question": question,
#             "answer_en": english_answer,
#             "data_type": data_type
#         }

#         # Translate to Arabic if requested
#         arabic_answer = None
#         if language == "ar":
#             try:
#                 arabic_answer = GoogleTranslator(source='en', target='ar').translate(english_answer)
#                 result["answer_ar"] = arabic_answer
#             except Exception as e:
#                 st.warning(f"Translation failed: {str(e)}")

#         # Generate TTS
#         text_for_tts = arabic_answer if language == "ar" and arabic_answer else english_answer
#         tts_lang = "ar" if language == "ar" else "en"

#         audio_bytes = io.BytesIO()
#         tts = gTTS(text=text_for_tts, lang=tts_lang)
#         tts.save(audio_bytes)
#         audio_bytes.seek(0)

#         result["audio_bytes"] = audio_bytes

#         return result
#     except requests.exceptions.RequestException as e:
#         st.error(f"Ollama API error: {str(e)}")
#         return None
#     except Exception as e:
#         st.error(f"Error generating response: {str(e)}")
#         return None

# # Streamlit UI
# st.title("Universal Document Assistant")

# # File Upload
# uploaded_file = st.file_uploader(
#     "Choose an Excel or PDF file",
#     type=['xlsx', 'xls', 'pdf'],
#     help="Upload a file to start asking questions."
# )

# if uploaded_file is not None:
#     file_content = uploaded_file.read()
#     filename = uploaded_file.name

#     if st.button("Process File"):
#         if filename.lower().endswith(('.xlsx', '.xls')):
#             processed_data = process_excel(file_content, filename)
#         else:
#             processed_data = process_pdf(file_content, filename)

#         if processed_data:
#             st.session_state.current_data = processed_data
#             st.success(f"File processed: {filename}")
#             if processed_data["type"] == "excel":
#                 st.info(f"Rows: {processed_data['rows']}, Columns: {', '.join(processed_data['columns'])}")
#             else:
#                 st.info(f"Pages: {processed_data['pages']}, Text Length: {processed_data['text_length']}")

# # Question Asking
# if st.session_state.current_data is not None:
#     st.header("Ask a Question")
#     question = st.text_area("Your question:", height=100)
#     language = st.selectbox("Language", ["en", "ar"], format_func=lambda x: "English" if x == "en" else "Arabic")

#     if st.button("Ask"):
#         if question.strip():
#             with st.spinner("Generating answer..."):
#                 result = ask_question(question, language, st.session_state.current_data)
#                 if result:
#                     if language == "ar" and "answer_ar" in result:
#                         st.write("**Answer (Arabic):**")
#                         st.write(result["answer_ar"])
#                     else:
#                         st.write("**Answer (English):**")
#                         st.write(result["answer_en"])

#                     st.audio(result["audio_bytes"], format="audio/mp3")
#         else:
#             st.warning("Please enter a question.")
# else:
#     st.info("Please upload and process a file first.")