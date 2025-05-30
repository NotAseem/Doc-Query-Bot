import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from PyPDF2 import PdfReader
import os
import shutil
import argparse
import google.generativeai as genai
import uuid
from cache_manager import get_cached_response, cache_response, get_similar_question, clear_cache
from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()
    
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
You are a helpful AI assistant. Your task is to answer questions based on the provided context. 
If you cannot find a clear answer in the context, say so.

Context:
{context}

Question: {question}

Instructions:
1. Answer based on the provided context
2. Be clear and informative in your response
3. Use the context to provide a helpful response
4. Don't mention document numbers in response
Answer:"""

# Helper Functions
def process_pdf(file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text, file.name  # Return both text and filename

def split_text_into_chunks(text):
    """Splits extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    documents = [Document(page_content=text)]
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks, filename):
    """Add chunks of text to the Chroma database."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Generate unique IDs for chunks using UUID
    unique_id = str(uuid.uuid4())
    for idx, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"{filename}_{unique_id}_chunk_{idx}"
        chunk.metadata["source"] = filename

    # Get existing document IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    # Add only new documents
    new_chunks = [
        chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    return db

def query_rag(query_text: str, db):
    """Query the RAG system and return the response."""
    # Check cache first
    cached_response = get_cached_response(query_text)
    if cached_response:
        content = cached_response.get('answer', str(cached_response))
        for char in content:
            yield char
        return

    similar_q = get_similar_question(query_text)
    if similar_q:
        _, answer = similar_q
        for char in str(answer):
            yield char
        return

    results = db.similarity_search_with_score(query_text, k=10)
    context_parts = [doc.page_content for doc, _ in results]
    context_text = "\n\n".join(context_parts)

    system_prompt = "You are a helpful AI assistant. Answer based only on the provided context."
    user_prompt = f"""Context:\n{context_text}\n\nQuestion: {query_text}\n\nAnswer:"""

    try:
        response_text = ""
        response = model.generate_content(user_prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                response_text += chunk.text
                yield chunk.text

        if response_text:
            cache_response(query_text, response_text)

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        yield f"Error: {str(e)}"

def clear_qa_cache():
    """Clear the QA cache contents."""
    clear_cache()
    return "Cache cleared successfully!"

# Streamlit Interface
def main():
    # CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args, unknown = parser.parse_known_args()
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )



    # Streamlit UI
    st.title("PDF Querying with RAG System")

    try:
        existing_items = db.get(include=[])
        has_documents = len(existing_items["ids"]) > 0
    except ValueError:
        # If collection is not initialized, treat as empty
        has_documents = False

    # Upload PDF
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            for uploaded_file in uploaded_files:
                pdf_text, filename = process_pdf(uploaded_file)
                chunks = split_text_into_chunks(pdf_text)
                db = add_to_chroma(chunks, filename)
        st.success(f"Successfully processed {len(uploaded_files)} file(s)!")

        # Query Input
        #query = st.text_input("Enter your query:")
        # if query:
        #     st.write("Searching for answers...")
        #     response, sources = query_rag(query, db)

        #     # Display Results
        #     st.write("### Response")
        #     st.write(response.content)

        #     st.write("### Sources")
        #     st.write(sources)
    if has_documents or uploaded_files is not None:
        # Initialize session state for toggle if not exists
        if 'show_docs' not in st.session_state:
            st.session_state.show_docs = False

        # Toggle button
        if st.button("📂 View Documents in Database"):
            st.session_state.show_docs = not st.session_state.show_docs

        # Show documents only if toggle is on
        if st.session_state.show_docs:
            st.subheader("📂 Files in Database")
            
            # --- Show existing documents ---
            try:
                # Get all documents and their metadata
                all_docs = db.get()
                if all_docs and all_docs['metadatas']:
                    # Create a set of unique filenames from the source metadata
                    unique_files = set()
                    for metadata in all_docs['metadatas']:
                        if 'source' in metadata:
                            unique_files.add(metadata['source'])
                    
                    # Display each unique filename
                    for filename in unique_files:
                        st.markdown(f"- 📄 `{filename}`")
                else:
                    st.info("No files found in the database.")
            except Exception as e:
                st.error(f"Error fetching files: {e}")

        st.subheader("🔍 Ask a Question")
        query = st.text_input("Enter your query:")
        if query:
            with st.spinner(""):
                # Create an empty container for the response
                response_container = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk in query_rag(query, db):
                    try:
                        # Handle different response formats
                        if isinstance(chunk, dict):
                            content = chunk.get('content', '')
                        elif hasattr(chunk, 'content'):
                            content = chunk.content
                        else:
                            content = str(chunk)
                        
                        if content:
                            full_response += content
                            # Format the response for display
                            formatted_response = full_response.replace('\n', '  \n')  # Add proper markdown line breaks
                            response_container.markdown(formatted_response, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing response chunk: {str(e)}")
                        continue
    else:
        st.info("Upload a PDF to enable querying.")

if __name__ == "__main__":
    main()