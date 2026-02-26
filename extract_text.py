import os
import json
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- PART 1: EXTRACT TEXT FROM PDFS ---
def extract_text_from_pdfs():
    pdf_folder = r"C:\ncert"
    pdf_filenames = [
        "NCERT-Class-12-Chemistry-Part-1.pdf",
        "NCERT-Class-12-Chemistry-Part-2.pdf"
    ]
    textbook_content = {}
    print("Starting PDF text extraction...")
    for filename in pdf_filenames:
        file_path = os.path.join(pdf_folder, filename)
        try:
            full_text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            book_name = os.path.splitext(filename)[0]
            textbook_content[book_name] = full_text
            print(f"-> Extracted text from {filename}")
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
    return textbook_content

# --- PART 2: CHUNK THE EXTRACTED TEXT ---
def chunk_the_text(textbook_content):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print("\nStarting the chunking process...")
    for book_name, text in textbook_content.items():
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_with_metadata = {
                'text_content': chunk,
                'metadata': {'source': book_name, 'chunk_number': i + 1}
            }
            all_chunks.append(chunk_with_metadata)
        print(f"-> Created {len(chunks)} chunks for {book_name}")
    return all_chunks

# --- PART 3: CREATE EMBEDDINGS AND VECTOR DATABASE ---
def create_vector_database(chunks_with_metadata):
    model_name = "all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    documents = []
    for chunk_data in chunks_with_metadata:
        doc = Document(
            page_content=chunk_data['text_content'],
            metadata=chunk_data['metadata']
        )
        documents.append(doc)

    print(f"\nCreating vector database from {len(documents)} documents...")
    vector_db = Chroma.from_documents(
        documents=documents, 
        embedding=embedding_model,
        persist_directory="./chroma_db" 
    )
    print("-> Vector database created and saved successfully!")
    return vector_db

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    extracted_data = extract_text_from_pdfs()
    chunked_data = chunk_the_text(extracted_data)
    vector_database = create_vector_database(chunked_data)
    print("\nSetup complete. Your 'chroma_db' folder is ready.")