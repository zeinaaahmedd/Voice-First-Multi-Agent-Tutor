import requests
import math

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b-instruct"


class SummaryQuizAgent:
    def __init__(self):
        self.params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
            "num_predict": 1200,
            "seed": 42
        }

        self.chunk_prompt = self.load("chunk_summary_prompt.txt")
        self.reduce_prompt = self.load("reduce_summary_prompt.txt")
        self.quiz_prompt = self.load("quiz_prompt.txt")

    # ---------------- UTIL ----------------
    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def call(self, messages, temperature=None, max_tokens=None):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            **self.params
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens:
            payload["num_predict"] = max_tokens

        r = requests.post(OLLAMA_URL, json=payload)
        return r.json()["message"]["content"]

    # ---------------- CHUNKING ----------------
    def chunk_text(self, text, chunk_size=1200):
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))

        return chunks

    # ---------------- MAP STAGE ----------------
    def summarize_chunk(self, chunk):
        messages = [
            {"role": "system", "content": self.chunk_prompt},
            {"role": "user", "content": chunk}
        ]

        return self.call(messages, temperature=0.2, max_tokens=400)

    def map_stage(self, chunks):
        return [self.summarize_chunk(c) for c in chunks]

    # ---------------- REDUCE STAGE ----------------
    def reduce_stage(self, partial_summaries):
        combined = "\n\n".join(partial_summaries)

        messages = [
            {"role": "system", "content": self.reduce_prompt},
            {"role": "user", "content": combined}
        ]

        return self.call(messages, temperature=0.2, max_tokens=800)

    # ---------------- QUIZ ----------------
    def generate_quiz(self, final_summary):
        messages = [
            {"role": "system", "content": self.quiz_prompt},
            {"role": "user", "content": final_summary}
        ]

        return self.call(messages, temperature=0.3, max_tokens=1200)

    # ---------------- PIPELINE ----------------
    def run(self, lecture_text):
        chunks = self.chunk_text(lecture_text)

        print(f"Chunks created: {len(chunks)}")

        partial_summaries = self.map_stage(chunks)

        final_summary = self.reduce_stage(partial_summaries)

        quiz = self.generate_quiz(final_summary)

        return {
            "chunks": len(chunks),
            "summary": final_summary,
            "quiz": quiz
        }

if __name__ == "__main__":
    summaryquizagent = SummaryQuizAgent()

    with open("lecture_sample.txt", "r", encoding="utf-8") as f:
        lecture = f.read()

    result = summaryquizagent.run(lecture)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY\n")
    print(result["summary"])

    print("\n" + "=" * 60)
    print("QUIZ\n")
    print(result["quiz"])