# RAG PDF Prototype

This is a **prototype** Retrieval-Augmented Generation (RAG) app that allows users to ask questions about a PDF document and receive accurate answers. The app demonstrates how to combine PDF text extraction, document chunking, embeddings, and a language model to retrieve relevant information on-demand.  

## Features

- Upload a PDF file and extract its text using `pdfminer`.
- Split large texts into smaller chunks for better retrieval using `RecursiveCharacterTextSplitter`.
- Convert text chunks into vector embeddings using `OpenAIEmbeddings`.
- Store embeddings in a FAISS vector store for efficient semantic search.
- Retrieve relevant document snippets using a retriever and generate answers with a language model (`ChatOpenAI`).
- Interactive interface built with Streamlit.

## Tech Stack

- **Python**  
- **Streamlit** – Web interface for uploading PDFs and asking questions  
- **PDFMiner** – Text extraction from PDF  
- **LangChain** – Orchestrating RAG workflow  
- **OpenAI API** – Generating embeddings and responses  
- **FAISS** – Vector database for semantic search  
- **Langsmith** – Tracking, logging, and managing LangChain runs

## How to Use

1. Clone this repository.
2. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API keys in Streamlit secrets:  
   ```text
   OPENAI_API_KEY
   LANGCHAIN_API_KEY
   LANGCHAIN_TRACING_V2
   ```

4. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```
5. Upload a PDF file and start asking questions about its content.