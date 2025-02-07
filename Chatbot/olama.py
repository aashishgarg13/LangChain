
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
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

llm = Ollama(model=selected_model, temperature=temperature, num_predict=max_tokens)

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
    
    
# Multi-language support 
language_options = ["English", "Frence", "Spanish", "German"]
selected_language  = st.sidebar.selectbox("Select a language", language_options)
  
# Prompt template  with context and language

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", f"You are a helpful assistant. Respond in {selected_language}. Use the following context if provided: {context}"),
        ("user", "Question: {question}")
    ]
)
chain = prompt_template | llm | StrOutputParser()

# Streamlit UI 

st.title('LangChain Demo with Ollama')

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
input_text = st.text_input("Search the topics you want")

   
# Process user input 

if input_text:
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": input_text})
        
        # Generate response with progress bar
        with st.spinner("Generating response..."):
            start_time = time.time()
            response = chain.invoke({"question": input_text})
            end_time = time.time()
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Update analytics
        st.session_state.query_count += 1
        response_time = end_time - start_time
        
        # Display the latest response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Feedback mechanism
        feedback = st.radio("Was this response helpful?", ["Yes", "No"], horizontal=True)
        if feedback == "Yes":
            st.success("Thank you for your feedback!")
        elif feedback == "No":
            st.error("We're sorry to hear that. Please let us know how we can improve.")
        
        # Display analytics in sidebar
        st.sidebar.metric("Total Queries", st.session_state.query_count)
        st.sidebar.metric("Last Response Time", f"{response_time:.2f} seconds")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        