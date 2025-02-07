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
    """
    API endpoint to compare semantic similarity of two sentences.
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract sentences
        sentence1 = data.get('sentence1', '').strip()
        sentence2 = data.get('sentence2', '').strip()

        # Validate input
        if not sentence1 or not sentence2:
            return jsonify({
                "error": "Both sentences are required",
                "is_similar": False,
                "reasoning": "Input sentences cannot be empty."
            }), 400

        # Prepare prompt for Groq
        messages = [
            {
                "role": "system",
                "content": """You are a precise semantic similarity evaluator. 
Carefully analyze two sentences and determine if they convey the same core meaning.
Respond EXACTLY in this JSON format (do not include any other text):
{
    "is_similar": true/false,
    "reasoning": "A concise explanation of why the sentences are similar or different."
}
Focus on semantic meaning, not exact wording."""
            },
            {
                "role": "user",
                "content": f"Compare these two sentences:\n1. '{sentence1}'\n2. '{sentence2}'\n\nDo they mean the same thing?"
            }
        ]

        # Call Groq API
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.2
        )

        # Extract the result
        result_str = response.choices[0].message.content

        # Parse JSON
        try:
            result = json.loads(result_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON. Raw response: {result_str}")
            return jsonify({
                "error": "Failed to parse AI response",
                "is_similar": False,
                "reasoning": "Internal processing error occurred."
            }), 500

        # Return results
        return jsonify({
            "is_similar": result.get('is_similar', False),
            "reasoning": result.get('reasoning', 'No reasoning provided.')
        })

    except Exception as e:
        # Detailed error logging
        print("Unexpected error occurred:")
        print(traceback.format_exc())

        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "is_similar": False,
            "reasoning": "Server encountered an unexpected error during processing."
        }), 500


@app.route('/')
def serve_app():
    return app.send_static_file('index.html')

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
