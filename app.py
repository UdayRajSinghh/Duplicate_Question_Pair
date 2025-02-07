from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import json
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',  # Specify the static folder name
            static_url_path=''       # This makes the static folder accessible at root URL
            )
CORS(app)   # Enable CORS for all routes

# Initialize Groq client
client = Groq(
    api_key=os.getenv('GROQ_API_KEY') # Store your API key in .env file
)


@app.route('/api/compare-sentences', methods=['POST'])
def compare_sentences():
    try:
        data = request.get_json()
        sentence1 = data.get('sentence1', '').strip()
        sentence2 = data.get('sentence2', '').strip()

        if not sentence1 or not sentence2:
            return jsonify({
                "error": "Both sentences are required",
                "is_similar": False,
                "similarity_percentage": 0,
                "reasoning": "Input sentences cannot be empty."
            }), 400

        messages = [
            {
                "role": "system",
                "content": """You are a precise multilingual semantic similarity evaluator. 
Analyze two sentences and determine their semantic similarity.
Respond EXACTLY in this JSON format (do not include any other text):
{
    "is_similar": true/false,
    "similarity_percentage": <number between 0 and 100>,
    "reasoning": "A concise explanation of why the sentences are similar or different, including specific details about matching and differing concepts."
}
The similarity_percentage should reflect how close the meanings are:
- 100%: Identical meaning
- 75-99%: Very similar with minor differences
- 50-74%: Partially similar
- 25-49%: Mostly different with some common elements
- 0-24%: Completely different meanings"""
            },
            {
                "role": "user",
                "content": f"Compare these two sentences:\n1. '{sentence1}'\n2. '{sentence2}'\n\nAnalyze their similarity."
            }
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.2
        )

        result_str = response.choices[0].message.content

        try:
            result = json.loads(result_str)
            # Validate percentage is within bounds
            similarity_percentage = float(result.get('similarity_percentage', 0))
            if not 0 <= similarity_percentage <= 100:
                similarity_percentage = max(0, min(100, similarity_percentage))
            
            return jsonify({
                "is_similar": result.get('is_similar', False),
                "similarity_percentage": similarity_percentage,
                "reasoning": result.get('reasoning', 'No reasoning provided.')
            })
        except json.JSONDecodeError:
            print(f"Failed to parse JSON. Raw response: {result_str}")
            return jsonify({
                "error": "Failed to parse AI response",
                "is_similar": False,
                "similarity_percentage": 0,
                "reasoning": "Internal processing error occurred."
            }), 500

    except Exception as e:
        print("Unexpected error occurred:")
        print(traceback.format_exc())
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "is_similar": False,
            "similarity_percentage": 0,
            "reasoning": "Server encountered an unexpected error during processing."
        }), 500

@app.route('/')
def serve_app():
    return app.send_static_file('index.html')

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
