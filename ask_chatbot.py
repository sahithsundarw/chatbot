import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
load_dotenv()
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- INITIALIZATION ---
def initialize_chatbot():
    """Initializes all components required for the chatbot."""
    print("Initializing chatbot components...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    
    # Explicitly load the API key and pass it to the constructor
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
        
    # UPDATED: Changed model name from "gemini-pro" to a current version
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        google_api_key=google_api_key, 
        temperature=0.3, 
        convert_system_message_to_human=True
    )
    
    prompt_template = """
You are an expert assistant for students. Your task is to answer the user's question based *only* on the provided context from their textbook.
Do not use any external knowledge. If the information is not in the context, clearly state that the answer is not available in the provided text.

Context:
{context}

Question:
{input}

Answer:
"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("Chatbot is ready. Type 'quit' or 'exit' to end the session.\n")
    return rag_chain

def ask_question(chain, question):
    """Invokes the RAG chain to get an answer for a given question."""
    print("\nThinking...")
    response = chain.invoke({"input": question})
    
    print("\n--- Answer ---")
    print(response.get("answer", "Sorry, I could not generate an answer."))
    
    print("\n--- Sources ---")
    if response.get("context"):
        sources = set(doc.metadata.get('source', 'Unknown') for doc in response["context"])
        for source in sources:
            print(f"- {source}")
    else:
        print("No sources were used.")

# --- MAIN INTERACTIVE LOOP ---
if __name__ == "__main__":
    chatbot_chain = initialize_chatbot()
    
    while True:
        # This line prompts the user for input in the terminal
        user_input = input("You: ")
        
        # Check if the user wants to exit the chat
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        # Ask the question and get the answer
        ask_question(chatbot_chain, user_input)
        print("\n" + "="*50)
