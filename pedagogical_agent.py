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

# ── LangGraph & LangChain Imports ───────────────────────────────────────────
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ── OpenRouter (LLM — cloud, no local GPU needed) ───────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "qwen/qwen-2.5-7b-instruct"


# ── 1. Define LangGraph State ───────────────────────────────────────────────
class AgentState(TypedDict):
    # add_messages appends new messages to the list instead of overwriting them
    messages: Annotated[list[BaseMessage], add_messages]


class PedagogicalAgentGraph:
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

        # ── System prompt ────────────────────────────────────────────────
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
        
        # ── Compile LangGraph Pipeline ───────────────────────────────────
        self.graph = self._build_graph()

    def retrieve_knowledge(self, query: str, n_results: int = 3) -> str:
        """Embed the query locally with E5-large and retrieve top-N docs."""
        try:
            embedding = self.embedder.encode(
                f"query: {query}",
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

    # ── 2. LangGraph Node Function ──────────────────────────────────────────
    def call_model(self, state: AgentState) -> dict:
        """Processes the state, injects RAG context, and hits OpenRouter API."""
        
        # Get the latest question from the user
        last_user_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
        user_query = last_user_message.content
        
        # Retrieve context matching the user's latest input
        retrieved_context = self.retrieve_knowledge(user_query)

        # Build formatted context wrapper
        if retrieved_context:
            context_wrapper = (
                f"السياق التعليمي المتوفر من قاعدة البيانات:\n{retrieved_context}\n\n"
                f"موضوع الطالب او سؤاله الحالي: "
            )
        else:
            context_wrapper = (
                f"ملحوظة: مفيش معلومات متوفرة في قاعدة البيانات عن الموضوع ده.\n\n"
                f"موضوع الطالب او سؤاله الحالي: "
            )

        # Standardize message structures to build the full history payload
        payload_messages = [{"role": "system", "content": self.system_prompt}]
        
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                # Only inject the RAG wrapper onto the last message to avoid polluting history
                if msg == last_user_message:
                    payload_messages.append({"role": "user", "content": f"{context_wrapper}{msg.content}"})
                else:
                    payload_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                payload_messages.append({"role": "assistant", "content": msg.content})

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": payload_messages,
        }

        try:
            print(f"LOG: Calling OpenRouter ({OPENROUTER_MODEL}) with conversation history...")
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            ai_reply = response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            ai_reply = "خطأ: OpenRouter ما ردتش في الوقت المطلوب. جرب تاني."
        except requests.exceptions.ConnectionError:
            ai_reply = "خطأ: مش قادر اتصل بـ OpenRouter. تاكد من الانترنت."
        except requests.exceptions.HTTPError as e:
            ai_reply = f"خطأ HTTP: {e} — {response.text[:300]}"
        except (KeyError, ValueError) as e:
            ai_reply = f"خطأ في معالجة الرد: {e}"

        # Return the new AI message back to the LangGraph state
        return {"messages": [AIMessage(content=ai_reply)]}

    # ── 3. Assemble LangGraph Workflow ──────────────────────────────────────
    def _build_graph(self):
        builder = StateGraph(AgentState)
        
        # Add the model evaluation step as a node
        builder.add_node("agent", self.call_model)
        
        # Define execution flow
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        
        # Enable state persistence checkpointing (In-Memory Saver)
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  BASIRA - المعلم الذكي بالمصري (مع LangGraph Memory)")
    print(f"  LLM : {OPENROUTER_MODEL} (OpenRouter)")
    print("  RAG : intfloat/multilingual-e5-large (local CPU)")
    print("=" * 60 + "\n")

    tutor = PedagogicalAgentGraph()

    print("\nاكتب موضوع او سؤال عايز تتعلمه.")
    print("اكتب 'خروج' او 'exit' للخروج.\n")

    # Establish configuration specifying the active thread ID session
    config = {"configurable": {"thread_id": "session_1"}}

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

        print("\nLOG: Running LangGraph pipeline...\n")
        
        # Input state object with the current question
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Process changes through the graph instance
        output_state = tutor.graph.invoke(input_state, config=config)
        
        # Snag the final updated response message 
        final_response = output_state["messages"][-1].content

        print("\n" + "=" * 60)
        print("[BASIRA]:")
        print(final_response)
        print("=" * 60 + "\n")