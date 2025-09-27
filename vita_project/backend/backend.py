from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI
llm = OpenAI(
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Create conversation memory
memory = ConversationBufferMemory()

# Define prompt template
template = """You are VITA, an AI assistant specialized in the truss industry and internal company projects.
Your responses should be helpful, accurate, and focused on truss-related topics.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Initialize conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Get response from conversation chain
        response = conversation.predict(input=user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)