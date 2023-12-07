from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained model for text generation
generator = pipeline("text-generation")

@app.route("/")
def home():
    return "Virtual Assistant is up and running!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("query", "")

    # Use the pre-trained model to generate a response
    response = generate_response(user_query)

    return jsonify({"response": response})

def generate_response(user_query):
    # You can customize this function based on your NLP model or use case
    # For simplicity, using a text generation model here
    prompt = f"You: {user_query}\nAssistant:"
    response = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]

    return response

if __name__ == "__main__":
    app.run(debug=True)
