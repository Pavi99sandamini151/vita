from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return jsonify({'error': 'Please use POST method to send messages'})
    
    # Handle POST request
    try:
        user_message = request.json.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'})
        
        # Simple chatbot logic - you can replace this with your actual AI logic
        if user_message.lower() in ['hello', 'hi', 'hey']:
            bot_response = "Hello! Welcome to VITA. How can I help you today?"
        elif user_message.lower() in ['how are you', 'how are you?']:
            bot_response = "I'm doing great! Thanks for asking. What can I assist you with?"
        elif user_message.lower() in ['bye', 'goodbye', 'see you']:
            bot_response = "Goodbye! Have a great day!"
        elif 'vita' in user_message.lower():
            bot_response = "VITA is here to help you! What would you like to know?"
        else:
            bot_response = f"I received your message: '{user_message}'. How can I help you further?"
        
        return jsonify({'response': bot_response})
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
