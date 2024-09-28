import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API key for Groq
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it and restart the application.")
    st.stop()

# Initialize session state
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "vector" not in st.session_state:
    st.session_state.embeddings = initialize_embeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq Demo")

# Initialize the ChatGroq model
@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=groq_api_key,
        model_name="Llama-3.2-90b-Text-Preview",
    )

llm = get_llm()

# Create a prompt template
prompt_template = ChatPromptTemplate.from_template(
"""
Answer the question given the provided context only.
Remember to give the most accurate answer based on the context that is given.
<context>
{context}
</context>
Question: {input}
"""
)

# Set up document chain
@st.cache_resource
def get_retrieval_chain():
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

retrieval_chain = get_retrieval_chain()

# User input
prompt = st.text_input("Enter your prompt here: ")

if prompt:
    with st.spinner("Generating response..."):
        start = time.time()
        response = retrieval_chain.invoke({"input": prompt})
        end = time.time()
        
    st.write(f"Response Time: {end - start:.2f} seconds")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(f"Document {i + 1}:")
            st.write(doc.page_content)
            st.write("-----------------------------------")