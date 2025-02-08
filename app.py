import os
from flask import Flask, request, jsonify

# Import LangChain components
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain

# Import Transformers pipeline for a free model
from transformers import pipeline

# Create a free LLM pipeline using a model from Hugging Face.
# Here we use "google/flan-t5-small" for text-to-text generation.
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    max_length=256  # Adjust as needed
)
llm = HuggingFacePipeline(pipeline=pipe)

def load_documents():
    """
    Load documents from the target URL using LangChain's URL loader.
    """
    url = "https://brainlox.com/courses/category/technical"
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {url}")
    return documents

def create_vector_store(documents):
    """
    Create a vector store by generating free embeddings for the documents.
    """
    # Using HuggingFaceEmbeddings with a free model (all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("Vector store created with document embeddings.")
    return vectorstore

def create_chatbot(vectorstore):
    """
    Create a conversational retrieval chain that uses the vector store as a retriever.
    """
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    print("Chatbot (conversational chain) is ready.")
    return conversation_chain

# Run the extraction and set up the chatbot
print("Starting document extraction...")
documents = load_documents()
vectorstore = create_vector_store(documents)
conversation_chain = create_chatbot(vectorstore)

# Create a Flask app for the RESTful API
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint to handle chat messages.
    Expects a JSON payload with:
    {
        "message": "Your question here",
        "chat_history": []  // Optional conversation history
    }
    Returns a JSON response with the chatbot's answer.
    """
    data = request.get_json(force=True)
    user_input = data.get("message", "")
    chat_history = data.get("chat_history", [])
    if not user_input:
        return jsonify({"error": "No message provided."}), 400

    # Get the answer from the conversational chain.
    result = conversation_chain({"question": user_input, "chat_history": chat_history})
    response = result.get("answer", "Sorry, I couldn't generate an answer.")
    return jsonify({"response": response})

if __name__ == "__main__":
    # Run the Flask app (default localhost:5000)
    app.run(debug=True)
