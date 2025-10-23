from pdfminer.high_level import extract_text
import streamlit as st
import langchainhub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langsmith import Client
import os

client = Client()



try:
    import streamlit as st
    IS_STREAMLIT = True
except ImportError:
    IS_STREAMLIT = False

RUNNING_ON_STREAMLIT = IS_STREAMLIT and hasattr(st, "secrets") and st.secrets is not None
RUNNING_ON_STREAMLIT_CLOUD = IS_STREAMLIT and os.environ.get("STREAMLIT_SERVER_HOST") is not None

print(RUNNING_ON_STREAMLIT_CLOUD)

if RUNNING_ON_STREAMLIT_CLOUD:
    # Running in Streamlit
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
    os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
    print("Environment: Streamlit")
else:
    # Running locally
    load_dotenv(dotenv_path=".env", override=True)
    print("Environment: Local")

llm = ChatOpenAI(model="gpt-4o-mini")

st.title("RAG app for PDF")
st.write("Ask questions about your PDF and get accurate answers with Retrieval Augmented Generation (RAG) app.")

uploaded_file = st.file_uploader("Choose a file", "pdf")


if uploaded_file is not None:
    text = extract_text(uploaded_file)

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Convert the extracted text into a Document object
    docs = [Document(page_content=text)]  

    # Split the document into smaller chunks
    splits = text_splitter.split_documents(docs) 

    # chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = client.pull_prompt("rlm/rag-prompt")

    def format_docs(text):
        return text

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.write("Completed processing the file. Now ask me anything about it!")
    # st.write(rag_chain.invoke("Give me a brief of the text"))

    prompt = st.text_input("Enter your prompt")
    if prompt:
        response = rag_chain.invoke(prompt)

        print(response)
        st.write(response)

else:
    st.info("Please upload a PDF file to extract text.")
