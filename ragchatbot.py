import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.header("My First ChatBot")


with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

if file is not None:
    with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
      

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)

    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        openai_api_key = OPENAI_API_KEY
    )

    vector_store = FAISS.from_texts(chunks,embeddings)

