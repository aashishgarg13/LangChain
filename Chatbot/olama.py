from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama  # ✅ Correct import
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key is set
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if not langchain_api_key:
    st.error("LANGCHAIN_API_KEY is missing. Please check your .env file.")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# **Prompt Template**
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# **Streamlit UI**
st.title('LangChain Demo with LLAMA3 API')
input_text = st.text_input("Search the topics you want")

# **Ollama LLM Instance**
llm = Ollama(model="llama3.2:latest")  # ✅ Correct

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# **Process User Input**
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
