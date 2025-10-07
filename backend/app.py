from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Configure URL
WEBSITE_URL = "https://en.wikipedia.org/wiki/Sherlock_Holmes"

app = Flask(__name__)
CORS(app)

class KnowledgeBase:
    def __init__(self):
        self.index = None
        self.texts = []
        self.embeddings = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_document(self, text):
        # Split text into smaller chunks
        chunks = [text[i:i+512] for i in range(0, len(text), 384)]
        self.texts.extend(chunks)
        
        # Get embeddings for chunks
        new_embeddings = self.model.encode(chunks)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Create or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(new_embeddings.shape[1])
        self.index.add(new_embeddings)
        print(f"Added {len(chunks)} chunks to knowledge base")

# Initialize knowledge base and QA model
kb = KnowledgeBase()
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def extract_text_from_website(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'head']):
            element.decompose()
            
        # Get main content
        main_content = soup.find('div', {'id': 'mw-content-text'})
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
            
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from website: {e}")
        return None

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if kb.index is None:
            return jsonify({'error': 'Knowledge base not initialized'}), 400
            
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Search relevant chunks
        query_vector = kb.model.encode([user_message])
        D, I = kb.index.search(query_vector.astype('float32'), k=3)
        context = "\n\n".join([kb.texts[i] for i in I[0]])
        
        # Get answer
        answer = qa_model(question=user_message, context=context)
        
        return jsonify({
            'response': answer['answer'],
            'confidence': float(answer['score'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def initialize_knowledge_base():
    try:
        print(f"Fetching content from: {WEBSITE_URL}")
        text = extract_text_from_website(WEBSITE_URL)
        
        if text:
            kb.add_document(text)
            print(f"Website content processed: {len(text)} characters")
            return True
        else:
            print("Failed to extract content from website")
            return False
            
    except Exception as e:
        print(f"Error initializing knowledge base: {type(e).__name__} - {str(e)}")
        return False
        
if __name__ == '__main__':
    print("Initializing knowledge base...")
    if initialize_knowledge_base():
        print("Knowledge base initialized successfully")
        app.run(port=5001, debug=True)
    else:
        print("Failed to initialize knowledge base")