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
import sys

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
        self.model = None

    def initialize_model(self):
        """Initialize the sentence transformer model"""
        if self.model is None:
            print("Loading sentence transformer model...")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Sentence transformer loaded")
            except Exception as e:
                print(f"✗ Error loading sentence transformer: {e}")
                raise

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
        self.index.add(new_embeddings.astype('float32'))
        print(f"✓ Added {len(chunks)} chunks to knowledge base")

# Global variables
kb = KnowledgeBase()
qa_model = None
initialization_status = {"status": "not_started", "message": "Not initialized"}

def extract_text_from_website(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print(f"Fetching URL: {url}")
        response = requests.get(url, headers=headers, timeout=30)
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
        
    except requests.Timeout:
        print("✗ Timeout error: Could not fetch website (timeout after 30s)")
        return None
    except requests.RequestException as e:
        print(f"✗ Network error fetching website: {e}")
        return None
    except Exception as e:
        print(f"✗ Error extracting text from website: {e}")
        return None

def initialize_knowledge_base():
    global qa_model, initialization_status
    try:
        print("\n" + "="*60)
        print("INITIALIZING KNOWLEDGE BASE")
        print("="*60)
        
        # Step 1: Fetch website content
        initialization_status = {"status": "initializing", "message": "Fetching website content..."}
        print("Step 1/4: Fetching website content...")
        text = extract_text_from_website(WEBSITE_URL)
        
        if not text:
            initialization_status = {"status": "error", "message": "Failed to fetch website content"}
            print("✗ Failed to extract content from website")
            return False
        
        print(f"✓ Fetched {len(text)} characters")
        
        # Step 2: Initialize sentence transformer
        initialization_status = {"status": "initializing", "message": "Loading sentence transformer..."}
        print("\nStep 2/4: Loading sentence transformer model...")
        print("(This will download ~80MB on first run)")
        kb.initialize_model()
        
        # Step 3: Process document
        initialization_status = {"status": "initializing", "message": "Processing document..."}
        print("\nStep 3/4: Processing document into chunks...")
        kb.add_document(text)
        
        # Step 4: Load QA model
        initialization_status = {"status": "initializing", "message": "Loading QA model..."}
        print("\nStep 4/4: Loading QA model...")
        print("(This will download ~250MB on first run)")
        qa_model = pipeline(
            'question-answering', 
            model='distilbert-base-cased-distilled-squad',
            device=-1  # Use CPU
        )
        print("✓ QA model loaded")
        
        initialization_status = {"status": "ready", "message": "Chatbot ready!"}
        print("\n" + "="*60)
        print("✓✓✓ KNOWLEDGE BASE INITIALIZED SUCCESSFULLY ✓✓✓")
        print("="*60 + "\n")
        return True
            
    except KeyboardInterrupt:
        initialization_status = {"status": "error", "message": "Initialization cancelled by user"}
        print("\n✗ Initialization cancelled")
        return False
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        initialization_status = {"status": "error", "message": error_msg}
        print(f"\n✗ Error during initialization: {error_msg}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify(initialization_status)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if initialization_status["status"] != "ready":
            return jsonify({
                'error': 'Chatbot is still initializing. Please wait and try again.',
                'status': initialization_status
            }), 503
            
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"\n📩 Question: {user_message}")
        
        # Search relevant chunks
        query_vector = kb.model.encode([user_message])
        D, I = kb.index.search(query_vector.astype('float32'), k=3)
        
        # Get relevant context
        relevant_chunks = [kb.texts[i] for i in I[0] if i < len(kb.texts)]
        context = "\n\n".join(relevant_chunks)
        
        print(f"📚 Found {len(relevant_chunks)} relevant chunks")
        
        # Get answer
        answer = qa_model(question=user_message, context=context)
        
        print(f"💬 Answer: {answer['answer']} (confidence: {answer['score']:.2f})")
        
        return jsonify({
            'response': answer['answer'],
            'confidence': float(answer['score'])
        })
        
    except Exception as e:
        print(f"✗ Error processing question: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING CHATBOT BACKEND SERVER")
    print("="*60)
    print("NOTE: First run will download AI models (~330MB total)")
    print("This may take 5-10 minutes depending on your connection")
    print("="*60 + "\n")
    
    # Initialize in background thread
    import threading
    init_thread = threading.Thread(target=initialize_knowledge_base)
    init_thread.daemon = True
    init_thread.start()
    
    print("Starting Flask server on http://127.0.0.1:5001")
    print("Initialization running in background...\n")
    
    # Run Flask without reloader to avoid double initialization
    app.run(host='127.0.0.1', port=5001, debug=False)
