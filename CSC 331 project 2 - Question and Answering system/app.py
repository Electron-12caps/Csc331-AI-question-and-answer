from flask import Flask, render_template, request, jsonify
import os
import re
import textwrap

try:
    import openai
except ImportError:
    openai = None

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def preprocess(text: str) -> dict:
    original = text.strip()
    lower = original.lower()
    no_punct = re.sub(r"[^0-9a-zA-Z\s]", "", lower)
    tokens = [t for t in no_punct.split() if t]
    cleaned = " ".join(tokens)
    return {
        "original": original,
        "lower": lower,
        "no_punct": no_punct,
        "tokens": tokens,
        "cleaned": cleaned,
    }


def build_prompt(processed: dict) -> str:
    prompt = textwrap.dedent(f"""
    You are an assistant that answers questions concisely and accurately.

    User's original question:
    {processed['original']}

    Processed (cleaned) question:
    {processed['cleaned']}

    Instructions:
    - Provide a short, direct answer (2–6 sentences).
    - Mention assumptions if needed.
    """)
    return prompt


def call_openai_chat(prompt: str) -> str:
    if openai is None:
        raise RuntimeError("openai package not installed. Run: pip install openai>=1.0.0")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.form or request.json
    question = data.get("question", "")

    processed = preprocess(question)
    prompt = build_prompt(processed)

    try:
        llm_response = call_openai_chat(prompt)
    except Exception as e:
        llm_response = f"[Error calling LLM API: {e}]"

    return jsonify({
        "processed": processed,
        "llm_response": llm_response,
        "final_answer": llm_response,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)