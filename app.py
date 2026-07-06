# Required Libraries
import os
from pyexpat import model
from dotenv import load_dotenv

import streamlit as st

from pypdf import PdfReader
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.genai import types 

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()
temp = 0.5

# Gemini API Key Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
client = genai.Client()

PROMPT_TEMPLATE = """
You are StudyAI, an AI study assistant that helps users understand documents they upload.

- Use only the information found in the context below.
- Do not use outside knowledge or make assumptions.
- If the answer is not contained in the context, reply exactly:
  "I don't know based on the provided documents."
- If multiple context passages are relevant, combine the information into a single coherent answer.
- Be concise but complete.
- Use bullet points when they improve readability.

========================
CONTEXT
========================
{relevant_passage}

========================
QUESTION
========================
{query}

========================
ANSWER
========================
"""

# Function: Loads Data using PyPDF2
def load_pdf(file):
    # Load the PDF file using PdfReader
    reader = PdfReader(file)
    # Extract text from each page
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    
    return text

# Function: Splits data into chunks
def split_text(text: str):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return text_splitter.split_text(text)

# Function: Embeds Data using Gemini Embeddings API
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Use the Gemini Embeddings API to generate embeddings for the input documents
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=input,
            config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING"),
        )  

        return [embedding.values for embedding in result.embeddings]

# Function: Create a persistent Chroma client and collection
def create_chroma_db():
    chroma_client = chromadb.PersistentClient(path = "data/")
    try:
        chroma_client.delete_collection("chroma_collection")
    except:
        pass

    db = chroma_client.create_collection(name="chroma_collection", embedding_function=GeminiEmbeddingFunction())
    return db

# Function: Stores Data in ChromaDB
def add_to_db(db, documents, file_name=""):

    for i, doc in enumerate(documents):
        db.add(documents=[doc], 
               ids = [f"{file_name}_{i}_{hash(doc)}"],
               metadatas=[{
                   "source": file_name,
                   "chunk": i
                   }]
                )

    return db

# Retrieving Relevant Passages from ChromaDB
def get_relevant_passage(query, db, n_results=3):
    results = db.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    return documents, metadatas

# Generate response using Gemini API
def generate_response(db, query, n_results=3):

    # Create a prompt using the relevant passages from the ChromaDB
    relevant_passages, metadatas = get_relevant_passage(query, db, n_results)
    context = ""

    for passage, metadata in zip(relevant_passages, metadatas):
        context += (
            f"Source: {metadata['source']} | "
            f"Chunk: {metadata['chunk']}\n"
            f"{passage}\n\n"
        )
    prompt = PROMPT_TEMPLATE.format(query=query, relevant_passage=context)

    # Call the Gemini API to generate a response based on the prompt
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temp)
    )

    return response.text, relevant_passages, metadatas

# ------- Streamlit App --------

# Configure Streamlit page
st.set_page_config(
    page_title="StudyAI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StudyAI")
st.caption("Your AI-powered Retrieval-Augmented Study Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content":"Hi! I'm your study assistant, what can I do for you today?"
        }
    ]

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = create_chroma_db()

if "total_files" not in st.session_state:
    st.session_state.total_files = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "file_names" not in st.session_state:
    st.session_state.file_names = []

db = st.session_state.db

### Knowledge Base ###
with st.sidebar:
    # -- Add section to upload plain text too --

    st.subheader("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type="pdf")

    index_clicked = st.button("📥 Index Documents")
    clear_clicked = st.button("🗑 Clear Knowledge Base")

    if clear_clicked:
        st.session_state.db = create_chroma_db()
        st.session_state.total_files = 0
        st.session_state.total_chunks = 0
        st.session_state.file_names = []
        st.success("Knowledge base cleared.")
        st.rerun()

    if uploaded_files and index_clicked:
        
        with st.spinner("Indexing documents..."):
            for file in uploaded_files:
                text = load_pdf(file)
                chunks = split_text(text)

                file_metadata = {
                    "name": file.name,
                    "type": file.type,
                    "size": file.size
                }
                
                add_to_db(db, chunks, file_metadata["name"])
                st.session_state.total_files +=1
                st.session_state.total_chunks += len(chunks)
                st.session_state.file_names.append(file.name)

        st.success(f"✓ Indexed {st.session_state.total_files} documents")
        st.success(f"✓ Created {st.session_state.total_chunks} searchable chunks")

    st.divider()
    st.write("Status")
    st.write("Documents indexed:")
    if db.count()>0:
        st.write("🟢 Ready")
        st.metric("Documents", st.session_state.total_files)
        st.metric("Chunks", st.session_state.total_chunks)   
    else:
        st.write("🔴 No documents indexed")
        st.write("Tip:  Make sure to index documents after upload.")

    st.divider()
    st.write("Uploaded files: ")
    if st.session_state.total_files > 0:
        for file_name in st.session_state.file_names:
            st.write(f"📄 {file_name}")
    else:
        st.write("No files currently in database")
        
### Chat ###
st.divider()
st.subheader("💬 Chat")
def display_messages():
    # Display all messages in the chat history
    for msg in st.session_state.messages:
        author = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(author):
            st.write(msg["content"])

# Display existing messages
display_messages()

# Handle new user input
user_input = st.chat_input("Ask a question about your uploaded documents...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Show thinking indicator while processing
    with st.chat_message("assistant"):
        if db.count() == 0:
            st.warning("Please upload a document first.")
            st.stop()
        else:
            with st.spinner("Searching documents..."):
                # Call above backend function
                response, passages, metadatas = generate_response(db, user_input)

            # Replace the spinner with actual response
            st.write(response)

            # Add response to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            ### Show context used ###
            st.divider()
            st.subheader("Retrieved Context")

            for passage, metadata in zip(passages, metadatas):
                with st.expander(
                    f"{metadata['source']} • Chunk {metadata['chunk']}"
                ):
                    st.write(passage)