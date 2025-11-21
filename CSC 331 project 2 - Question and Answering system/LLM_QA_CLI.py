import os
import re
import textwrap

try:
    import openai
except ImportError:
    openai = None

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

    Original user question:
    {processed['original']}

    Cleaned question:
    {processed['cleaned']}

    Instructions:
    - Provide a helpful answer (2–6 sentences).
    - Mention assumptions if needed.
    - Keep the response concise.
    """)
    return prompt


def call_openai_chat(prompt: str) -> str:
    if openai is None:
        raise RuntimeError("openai package not installed. Run: pip install openai>=1.0.0")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def main():
    print("LLM Q&A CLI — Ask anything (Ctrl+C to exit).")
    try:
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                print("Please enter a valid question.")
                continue

            processed = preprocess(question)
            print("\n--- Preprocessed Question ---")
            print(f"Original : {processed['original']}")
            print(f"Cleaned  : {processed['cleaned']}")
            print(f"Tokens   : {processed['tokens']}")

            prompt = build_prompt(processed)
            print("\nSending to LLM...")

            try:
                answer = call_openai_chat(prompt)
            except Exception as e:
                print(f"Error calling LLM API: {e}")
                answer = (
                    "[Fallback] Could not reach LLM API. "
                    f"Cleaned question: '{processed['cleaned']}'"
                )

            print("\n--- Answer ---")
            print(answer)

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()