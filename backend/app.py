from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from services.confluence_service import ConfluenceService
import os
from dotenv import load_dotenv
import uuid
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize models
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
confluence_service = ConfluenceService()

# Initialize knowledge base
def initialize_knowledge_base():
    pages = confluence_service.get_space_content()
    texts = []
    embeddings = []
    
    for page in pages:
        content = f"Title: {page['title']}\n\nContent: {page['content']}"
        chunks = [content[i:i+512] for i in range(0, len(content), 384)]  # 128 overlap
        texts.extend(chunks)
        chunk_embeddings = sentence_model.encode(chunks)
        embeddings.extend(chunk_embeddings)
    
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, texts

# Initialize knowledge base
index, texts = initialize_knowledge_base()

# Message store for MCP
message_store = {}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Create MCP message
        message_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        
        mcp_message = {
            'id': message_id,
            'timestamp': timestamp,
            'type': 'user_message',
            'content': user_message,
            'status': 'processing'
        }
        
        message_store[message_id] = mcp_message
        
        try:
            # Search knowledge base
            query_vector = sentence_model.encode([user_message])
            D, I = index.search(query_vector.astype('float32'), k=3)
            context = "\n\n".join([texts[i] for i in I[0]])
            
            # Get response using question-answering
            qa_response = qa_model(
                question=user_message,
                context=context
            )
            
            response = qa_response['answer']
            
            # Create response message
            response_id = str(uuid.uuid4())
            mcp_response = {
                'id': response_id,
                'timestamp': int(time.time() * 1000),
                'type': 'bot_response',
                'content': response,
                'status': 'completed',
                'reference_id': message_id,
                'context_used': context,
                'confidence': float(qa_response['score'])
            }
            
            message_store[response_id] = mcp_response
            mcp_message['status'] = 'completed'
            
            return jsonify({
                'message_id': message_id,
                'response_id': response_id,
                'response': response,
                'status': 'success'
            })
            
        except Exception as e:
            mcp_message['status'] = 'failed'
            mcp_message['error'] = str(e)
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ...existing code for get_message route...