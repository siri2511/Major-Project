import os
import openai
import openai
import pandas as pd

# === Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "api_key")
GLOSS_CSV_PATH = "/content/sign-spotter/output2/pred_glosses_confidences_filtered/th0.5_test_video_1.mp4.csv"
OUTPUT_SENTENCE_PATH = "/content/sign-spotter/output2/generated_sentences/sample_video_sentence.txt"

# === Set API Key ===
openai.api_key = OPENAI_API_KEY

# === Prompt Template ===
PROMPT_TEMPLATE = """
You are a helpful assistant designed to generate a sentence based on the list of words entered by the user. You need to strictly follow these rules:
1) The user will only give the list of English words separated by a space, you just need to generate a meaningful sentence from them.
2) Only provide a response containing the generated sentence. If you cannot create a English sentence then respond with “No Translation”.

Glosses: {}
"""

def generate_sentence(glosses):
    """
    Sends glosses to ChatGPT and returns the generated sentence.
    """
    prompt = PROMPT_TEMPLATE.format(" ".join(glosses))

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[{"role": "user", "content": prompt}],
          temperature=0.5,
          max_tokens=100
          )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] Failed to generate sentence: {e}")
        return "No Translation"

def main():
    # === Load glosses ===
    try:
        df = pd.read_csv(GLOSS_CSV_PATH, delimiter='|')
        if 'predicted_gloss' not in df.columns:
            raise ValueError("CSV must contain a 'predicted_gloss' column.")
        glosses = df['predicted_gloss'].dropna().unique().tolist()
    except Exception as e:
        print(f"[ERROR] Failed to load glosses: {e}")
        return

    # === Generate sentence ===
    sentence = generate_sentence(glosses)

    # === Output ===
    print("\nDetected Glosses:", " ".join(glosses))
    print("Generated Sentence:", sentence)

    # === Save result ===
    os.makedirs(os.path.dirname(OUTPUT_SENTENCE_PATH), exist_ok=True)
    with open(OUTPUT_SENTENCE_PATH, 'w', encoding='utf-8') as f:
        f.write(sentence)

if __name__ == "__main__":
    main()
