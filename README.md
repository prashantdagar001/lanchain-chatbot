# Project Title

Flask Chatbot with LangChain

## Description

This project is a Flask-based chatbot that utilizes LangChain components for document loading, embedding, and conversational retrieval. It allows users to interact with a chatbot through a RESTful API.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. The application will be available at `http://localhost:5000`.

## API Endpoints

### POST /chat

This endpoint handles chat messages.

**Request Body:**
```json
{
    "message": "Your question here",
    "chat_history": []  // Optional conversation history
}
```

**Response:**
```json
{
    "response": "Chatbot's answer here."
}
```

## Dependencies

- Flask
- LangChain
- Transformers
- FAISS
