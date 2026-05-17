import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── OpenRouter Configuration ────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL        = "meta-llama/llama-3.3-70b-instruct"

# ── LLM-as-a-Judge Prompt Rubric ───────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = (
    "You are an expert quality assurance evaluator and linguistic judge. Your job is to evaluate "
    "an educational chatbot named 'Basira' which teaches students in the Egyptian Arabic dialect.\n\n"
    "You will be given:\n"
    "1. The User's Query.\n"
    "2. The Provided RAG Context (if any).\n"
    "3. Basira's Response.\n\n"
    "Evaluate Basira's response strictly based on the following 4 criteria. "
    "Scale each metric between 2 and 4 (2 = Failed/Poor, 3 = Moderate, 4 = Perfect/Excellent):\n\n"
    
    "1. Dialect_and_Tone (Scale: 2-4):\n"
    "   - Score 4: Exclusively uses natural, warm, and clear Egyptian colloquial Arabic (e.g., 'يا بطل', 'بص يا سيدي').\n"
    "   - Score 2: Uses standard Arabic (Fusha) or sounds unnatural/stiff.\n\n"
    
    "2. Language_Purity (Scale: 2-4):\n"
    "   - Score 4: ZERO English words (not even technical terms spelled in English letters).\n"
    "   - Score 2: Contains one or more English words or Latin characters.\n\n"
    
    "3. Pedagogical_Structure (Scale: 2-4):\n"
    "   - Score 4: Explanations are broken down into short, clear, audio-friendly sentences based accurately on the context.\n"
    "   - Score 2: Long, dense walls of text, completely irrelevant explanations, or hallucinations.\n\n"
    
    "4. Engagement_Question (Scale: 2-4):\n"
    "   - Score 4: Concludes naturally with a simple, engaging check-for-understanding question in Egyptian Arabic.\n"
    "   - Score 2: No question asked at the end.\n\n"
    
    "CRITICAL OUTPUT FORMAT:\n"
    "You must return your output ONLY as a valid JSON object. Do not include introductory text or markdown prose outside the JSON block. "
    "Use the exact JSON schema below:\n"
    "{\n"
    "  \"scores\": {\n"
    "    \"dialect_and_tone\": int,\n"
    "    \"language_purity\": int,\n"
    "    \"pedagogical_structure\": int,\n"
    "    \"engagement_question\": int\n"
    "  },\n"
    "  \"rationale\": {\n"
    "    \"dialect_and_tone\": \"Brief reasoning for this score (max 15 words)\",\n"
    "    \"language_purity\": \"Brief reasoning for this score (max 15 words)\",\n"
    "    \"pedagogical_structure\": \"Brief reasoning for this score (max 15 words)\",\n"
    "    \"engagement_question\": \"Brief reasoning for this score (max 15 words)\"\n"
    "  },\n"
    "  \"passed\": bool\n"
    "}\n\n"
    "Set \"passed\" to true ONLY if all scores are 3 or 4. If any score is 2, set \"passed\" to false."
)

# ── Response Parser Utility ─────────────────────────────────────────────────
def parse_judge_response(response_text: str) -> dict:
    """
    Cleans up the LLM text output and extracts the raw JSON block safely.
    This guarantees we can parse the scores even if the model adds markdown formatting.
    """
    try:
        # If the model wrapped the response in a markdown code block ```json ... ```, strip it
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        return json.loads(response_text.strip())
    except Exception as e:
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_response": response_text
        }


# ── Core Evaluation Engine ──────────────────────────────────────────────────
def evaluate_response(user_query: str, rag_context: str, assistant_response: str) -> dict:
    """
    Formulates the user payload, sends it to Llama-3.3-70b-Instruct via OpenRouter,
    and returns a parsed dictionary containing the evaluation scores and rationales.
    """
    if not OPENROUTER_API_KEY:
        return {"error": "Missing OpenRouter API Key in judge environment."}

    # Construct the structural evaluation payload
    user_evaluation_payload = (
        f"### INPUT DATA FOR EVALUATION ###\n\n"
        f"1. User Query:\n\"{user_query}\"\n\n"
        f"2. Retrieved RAG Context:\n\"{rag_context if rag_context else 'No context provided.'}\"\n\n"
        f"3. Basira's Response:\n\"{assistant_response}\"\n\n"
        f"Please evaluate the response according to your system rubric instructions."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_evaluation_payload}
        ],
        "temperature": 0.1,  # Low temperature keeps grading deterministic and stable
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        
        raw_text = response.json()["choices"][0]["message"]["content"]
        
        # Process and parse into clean JSON
        return parse_judge_response(raw_text)

    except requests.exceptions.RequestException as e:
        return {"error": f"Network or HTTP error during evaluation: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}