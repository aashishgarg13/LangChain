from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st 
import os 
from dotenv import load_dotenv
import time 


# Load environment variables

load_dotenv()

# Initialize session state for chat history and analytics

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "quert_state" not in st.session_state:
    st.session_state.query_count = 0 
    
# Sidebar for model selections and advanced settings
model_options = ["llama3.2:latest", "llama2:latest", "other-ollama-model"]
selected_model = st.sidebar.selectbox("Select a model", model_options)
temperature= st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens =  st.sidebar.number_input("Max tokens", min_value=50, max_value=1000, value=250)

# Initialize ollama LLM
llm = ollama(model=selected_model, temperature=temperature , num_predict=max_tokens)

# File upload Ollama LLM

uploaded_file =  st.sidebar.file_uploader("Upload a file for contextual queries", type=["text", "pdf"])
context = ""
if uploaded_file:
    with open("temp_file", "wb") as f:
        f.write(uploaded_file.getvalue())
        
    loader = TextLoader("temp_file")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = texts[0].page_content if texts else "" 
    st.sidebar.write(f"Context loaded from file: {context[:100]}...")
    
