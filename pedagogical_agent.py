import os
import sys
import torch
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Prevent CUDA DLLs from loading — embedder runs on CPU, GPU stays free
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_KERAS"] = "1"

# Force UTF-8 console output so Arabic prints correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
import chromadb
from sentence_transformers import SentenceTransformer


# ── OpenRouter (LLM — cloud, no local GPU needed) ───────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "qwen/qwen-2.5-7b-instruct"


class PedagogicalAgent:
    def __init__(self):
        # ── Local embedding model on CPU ─────────────────────────────────
        print("LOG: Loading intfloat/multilingual-e5-large on CPU...")
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
        self.embedder.encode("warmup", show_progress_bar=False)  # JIT warmup
        print("LOG: Embedding model ready.")

        # ── ChromaDB ────────────────────────────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path="./rag_db")
        self.collection    = self.chroma_client.get_collection(name="content_arabic")
        print(f"LOG: Connected to 'content_arabic' ({self.collection.count()} documents).")

        # ── System prompt — Egyptian Arabic tutor persona (BASIRA) ──────
        self.system_prompt = (
            "أنت معلم وميسر تعليمي ذكي، صبور، ومشجع اسمك 'بصيرة'. "
            "مهمتك هي تعليم الطلاب باستخدام اللهجة المصرية العامية البسيطة والواضحة. "
            "التزم بالقواعد دي:\n"
            "1. ممنوع تماماً أي كلمة إنجليزية في ردك — كل حاجة بالعربي المصري.\n"
            "2. اتكلم بأسلوب مصري ودي (مثل: 'يا بطل'، 'بص يا سيدي'، 'تمام؟'، 'خد بالك').\n"
            "3. أسلوبك يكون تفاعلي ومشجع دايماً.\n"
            "4. اجعل جملك قصيرة ومرتبة — الطالب بيسمعك صوتياً فقط.\n"
            "5. بعد الشرح، اسأل سؤال بسيط باللهجة المصرية تتأكد إن الطالب فاهم."
        )

    # ──────────────────────────────────────────────────────────────────
    def retrieve_knowledge(self, query: str, n_results: int = 3) -> str:
        """Embed the query locally with E5-large and retrieve top-N docs."""
        try:
            embedding = self.embedder.encode(
                f"query: {query}",     # E5 requires this prefix
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            results   = self.collection.query(query_embeddings=[embedding], n_results=n_results)
            documents = results.get("documents", [[]])[0]

            if not documents:
                print("LOG: No relevant documents found in RAG.")
                return ""

            print(f"LOG: Retrieved {len(documents)} document(s) from RAG.")
            return "\n---\n".join(documents)

        except Exception as e:
            print(f"ERROR in RAG retrieval: {e}")
            return ""

    # ──────────────────────────────────────────────────────────────────
    def generate_response(self, user_query: str) -> str:
        """Retrieve RAG context then call OpenRouter (Qwen 7B) for the answer."""
        retrieved_context = self.retrieve_knowledge(user_query)

        if retrieved_context:
            user_content = (
                f"السياق التعليمي المتوفر من قاعدة البيانات:\n{retrieved_context}\n\n"
                f"موضوع الطالب او سؤاله: {user_query}"
            )
        else:
            user_content = (
                f"ملحوظة: مفيش معلومات متوفرة في قاعدة البيانات عن الموضوع ده.\n\n"
                f"موضوع الطالب او سؤاله: {user_query}"
            )

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_content},
            ],
        }

        try:
            print(f"LOG: Calling OpenRouter ({OPENROUTER_MODEL})...")
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            return "خطأ: OpenRouter ما ردتش في الوقت المطلوب. جرب تاني."
        except requests.exceptions.ConnectionError:
            return "خطأ: مش قادر اتصل بـ OpenRouter. تاكد من الانترنت."
        except requests.exceptions.HTTPError as e:
            return f"خطأ HTTP: {e} — {response.text[:300]}"
        except (KeyError, ValueError) as e:
            return f"خطأ في معالجة الرد: {e}"


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  BASIRA - المعلم الذكي بالمصري")
    print(f"  LLM : {OPENROUTER_MODEL} (OpenRouter)")
    print("  RAG : intfloat/multilingual-e5-large (local CPU)")
    print("=" * 60 + "\n")

    tutor = PedagogicalAgent()

    print("\nاكتب موضوع او سؤال عايز تتعلمه.")
    print("اكتب 'خروج' او 'exit' للخروج.\n")

    while True:
        try:
            user_input = input("[موضوع] اكتب هنا: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nمع السلامة!")
            break

        if not user_input:
            print("من فضلك اكتب موضوع او سؤال.\n")
            continue

        if user_input.lower() in ("خروج", "exit", "quit", "q"):
            print("مع السلامة!")
            break

        print("\nLOG: Running pipeline...\n")
        response = tutor.generate_response(user_query=user_input)

        print("\n" + "=" * 60)
        print("[BASIRA]:")
        print(response)
        print("=" * 60 + "\n")