import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = Flask(__name__)

PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize once at startup (loading the embedding model is expensive)
print("Loading embedding model and vector database...")
_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
_vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=_embedding_model)

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=google_api_key,
    temperature=0.3,
    convert_system_message_to_human=True,
)

_prompt = PromptTemplate.from_template("""
You are an expert assistant for students. Your task is to answer the user's question based *only* on the provided context from their textbook.
Do not use any external knowledge. If the information is not in the context, clearly state that the answer is not available in the provided text.

Context:
{context}

Question:
{input}

Answer:
""")

_retriever = _vector_db.as_retriever(search_kwargs={"k": 5})
_qa_chain = create_stuff_documents_chain(_llm, _prompt)
_rag_chain = create_retrieval_chain(_retriever, _qa_chain)
print("Chatbot ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        response = _rag_chain.invoke({"input": question})
        answer = response.get("answer", "Sorry, I could not generate an answer.")
        sources = list({
            doc.metadata.get("source", "Unknown")
            for doc in response.get("context", [])
        })
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
