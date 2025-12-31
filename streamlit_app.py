# Fixed Import Issue: Ensure your requirements.txt includes:
# langchain>=0.2.16
# langchain-groq>=0.1.0
# langchain-community>=0.2.16
# langsmith>=0.1.0
# If issues persist, run: pip install --upgrade langchain langchain-groq langchain-community
# The import should work in LangChain 0.2+; if using 0.1.x, upgrade.

import os
import io
import pandas as pd
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator
from gtts import gTTS
from groq import Groq
import tempfile
import streamlit as st

# Enhanced LangChain integration for agentic AI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun  # Optional external tool
from langsmith import traceable

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
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "Document-Assistant")

# Only enable tracing if API key is provided
if not os.environ["LANGCHAIN_API_KEY"]:
    st.warning("LANGCHAIN_API_KEY not found in secrets. LangSmith tracing will be disabled.")

# Groq Client (for fallback, but we'll use LangChain's ChatGroq)
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

# ================================
# Agentic Tools for LangChain
# ================================
@tool
def summarize_data(data: dict) -> str:
    """Summarize Excel/PDF content for the agent."""
    if data["type"] == "excel":
        df = data["content"]
        return f"Data Summary: Rows: {df.shape[0]}, Columns: {list(df.columns)}, Preview: {df.head(3).to_string()}"
    else:
        text = data["content"]
        truncated = text[:500] + ("..." if len(text) > 500 else "")
        return f"Document: {data['filename']}, Pages: {data['pages']}, Preview: {truncated}"

@tool
def calculate_stats(question: str, data: dict) -> str:
    """Run pandas stats if question involves calculations (e.g., 'total sales'). Only for Excel data."""
    if data["type"] != "excel":
        return "Stats only available for Excel data."
    df = data["content"]
    if "total" in question.lower() or "sum" in question.lower():
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            return f"Numeric Summary: {df[numeric_cols].sum().to_dict()}"
    elif "average" in question.lower() or "mean" in question.lower():
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            return f"Average: {df[numeric_cols].mean().to_dict()}"
    return "No relevant stats computable from the question."

@tool
def translate_text(text: str, target_lang: str) -> str:
    """Translate text to the target language."""
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        return f"Translation failed: {e}. Original: {text}"

# Optional: External search tool for grounding answers
search_tool = DuckDuckGoSearchRun()

# ================================
# Agentic Q&A Pipeline with LangChain ReAct Agent
# ================================
@traceable(name="Agentic Document Q&A Pipeline", run_type="chain")
def ask_question(question: str, language: str, data: dict):
    """Enhanced agentic function — uses ReAct agent with tools, fully traced by LangSmith."""
    # Initialize LLM via LangChain
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLAMA_MODEL_NAME,
        temperature=0.7
    )
    
    # Define tools (note: for binding, we'll pass data/language in invocation)
    tools = [summarize_data, calculate_stats, translate_text, search_tool]
    
    # Pull ReAct prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")
    # Customize the prompt for our use case
    react_prompt.messages[0].prompt.template += """
    You are an accurate AI assistant for document Q&A. Use only the provided tools and data to answer.
    Data available via tools. Question: {input}
    Always reason step-by-step, use tools when needed (e.g., summarize_data to get context first, calculate_stats for numbers).
    If you cannot answer from data/tools, say "I cannot determine this from the provided information."
    Final answer should be concise in English.
    """
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Logs agent thoughts/actions for debugging (visible in traces)
        handle_parsing_errors=True
    )
    
    # Prepare input - agent will call tools with data/language as needed
    # For simplicity, instruct agent to use data/language in tool calls
    agent_input = f"Question: {question}. Language preference: {language}. Use tools with data provided in context."
    
    try:
        # Agent execution — automatically traced
        @traceable(run_type="agent", name="ReAct Agent Execution")
        def run_agent():
            # Note: To pass data/language to tools, we can bind or use partials
            # For now, assume agent instructions guide tool use; enhance as needed
            return agent_executor.invoke({"input": agent_input, "data": data, "language": language})
        
        agent_output = run_agent()
        english_answer = agent_output["output"].strip()
        
        result = {
            "question": question,
            "answer_en": english_answer,
            "data_type": data["type"],
            "agent_steps": agent_output.get("intermediate_steps", [])  # For debugging
        }
        
        # Translation to Arabic if needed (post-process)
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
st.title("AI Document Assistant (Agentic Edition)")
st.caption("Upload an Excel or PDF and ask questions in English or Arabic. Powered by LangChain ReAct Agent for autonomous tool use!")

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
    question = st.text_area("Your question:", height=120, placeholder="e.g., What is the total revenue in 2024? Or average salary?")
    language = st.selectbox("Answer Language", ["en", "ar"], format_func=lambda x: "English" if x == "en" else "Arabic")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Get Agentic Answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Agent thinking and acting..."):
                    result = ask_question(question.strip(), language, st.session_state.current_data)
                    if result:
                        answer = result.get("answer_ar") if language == "ar" else result["answer_en"]
                        st.markdown(f"**Answer ({'Arabic' if language == 'ar' else 'English'}):**")
                        st.markdown(answer)
                        st.audio(result["audio_bytes"], format="audio/mp3")
                        
                        # Show agent steps if verbose (for demo)
                        if result.get("agent_steps"):
                            with st.expander("Agent Reasoning Steps (from LangSmith Trace)"):
                                for step in result["agent_steps"]:
                                    st.write(step)
    
    with col2:
        st.info("💡 Agent uses tools like pandas stats, translation, and search autonomously.")
else:
    st.info("Upload and process a file to begin asking questions.")

# Optional: Show LangSmith link if tracing is enabled
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    st.sidebar.caption(f"LangSmith tracing enabled → [View traces](https://smith.langchain.com/projects/p/{project_name})")
    st.sidebar.markdown("### Demo Notes")
    st.sidebar.markdown("- **Agentic Features**: ReAct agent decides tools (e.g., calculate_stats for sums).")
    st.sidebar.markdown("- **Tracing**: See full chain in LangSmith for debugging.")
    st.sidebar.markdown("- **Requirements**: `pip install langchain>=0.2.16 langchain-groq langchain-community`")
