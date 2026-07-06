# 📚 StudyAI

StudyAI is a Retrieval-Augmented Generation (RAG) application built with Streamlit, Google Gemini, and ChromaDB. It allows users to upload documents, index them into a vector database, and ask natural language questions whose answers are generated from the uploaded documents.

## Features

- 📄 Upload one or more PDF documents
- ✂️ Automatic document chunking
- 🔎 Semantic retrieval using vector embeddings
- 🤖 AI-powered question answering with Gemini
- 📚 Displays the retrieved context used to generate each response
- 💬 Interactive chat interface built with Streamlit

---

## RAG Pipeline

The application follows a standard Retrieval-Augmented Generation workflow:

1. **Document Upload**
   - Users upload one or more PDF files.

2. **Text Extraction**
   - Text is extracted from each page using **PyPDF**.

3. **Text Chunking**
   - Documents are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
   - Chunk size: **1000 characters**
   - Chunk overlap: **200 characters**

4. **Embedding Generation**
   - Each chunk is converted into a vector embedding using Google's **Gemini Embedding API** (`gemini-embedding-2`).

5. **Vector Storage**
   - Embeddings are stored in a **ChromaDB** collection along with metadata such as:
     - Source document
     - Chunk number

6. **Retrieval**
   - When the user submits a question, the query is embedded.
   - ChromaDB performs semantic similarity search to retrieve the most relevant chunks.

7. **Generation**
   - The retrieved chunks are inserted into a prompt.
   - **Gemini 3.1 Flash Lite** generates a response using only the retrieved context.

---

## Architecture

```
            PDF Upload
                 │
                 ▼
          Text Extraction
                 │
                 ▼
         Document Chunking
                 │
                 ▼
      Gemini Embedding Model
                 │
                 ▼
             ChromaDB
                 │
         Similarity Search
                 │
                 ▼
       Retrieved Context
                 │
                 ▼
      Gemini 3.1 Flash Lite
                 │
                 ▼
        Final Answer to User
```

---

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- LangChain Text Splitters
- PyPDF

---

## Embedding Model

- **Model:** `gemini-embedding-2`
- Used for generating dense vector representations of document chunks and user queries for semantic search.

---

## Vector Database

- **Database:** ChromaDB
- Stores document embeddings together with metadata.
- Performs similarity search to retrieve the most relevant document chunks during question answering.

---

## Generation Model

- **Model:** `gemini-3.1-flash-lite`
- Generates answers using the retrieved document context.

---

## Project Structure

```
StudyAI/
│
├── app.py
├── data/
├── requirements.txt
├── README.md
└── .env
```

---

## Future Improvements

- Support for DOCX, TXT, and Markdown files
- Persistent knowledge bases
- Citation highlighting
- Chat history with memory
- Better document management
- Adjustable retrieval parameters
- Enhanced Streamlit UI