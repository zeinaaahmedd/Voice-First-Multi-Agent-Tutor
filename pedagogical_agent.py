import os
import sys
import torch
import json                  # Added: To pretty-print the judge results
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

import chromadb
from sentence_transformers import SentenceTransformer

# ── Google GenAI SDK Import ─────────────────────────────────────────────────
from google import genai
from google.genai import types

# ── LangGraph & LangChain Imports ───────────────────────────────────────────
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ── Imported Judge Function ─────────────────────────────────────────────────
from judge import evaluate_response # Added: Connects this main loop to judge.py

# ── Direct Gemini API Initialization ────────────────────────────────────────
# The client automatically uses os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client()
GEMINI_MODEL = "gemini-2.5-flash"


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
            "أنت معلمة اسمك 'بصيرة'. مهمتك تشرحي المواضيع للأطفال بالعربي المصري الواضح والبسيط.\n\n"

            "قواعد لازم تتبعيها:\n"
            "1. ممنوع أي كلمة إنجليزية — كل حاجة بالعربي المصري فقط.\n"
            "2. اشرحي بالتفصيل: مش أقل من 6 جمل في الشرح.\n"
            "3. استخدمي المعلومات اللي في السياق التعليمي المقدم لك بدقة — لا تخترعي معلومات.\n"
            "4. قاعدة علمية مهمة: النباتات بتصنع أكلها بنفسها من ضوء الشمس عن طريق البناء الضوئي. "
            "البشر مش بياكلوا ضوء الشمس — ممنوع تقولي كده. لو بتشبهي النبات بالبشر, قولي: "
            "'بدل الأكل اللي بياكله البشر، النبات بيعمل أكله بنفسه من الضوء'.\n"
            "5. اتكلمي بأسلوب مصري دافي (مثل: 'يا بطل'، 'بص يا سيدي'، 'خد بالك').\n"
            "6. في آخر ردك، اسألي سؤال بسيط بالمصري عشان تتأكدي إن الطالب فاهم."
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
        """Processes the state, injects RAG context, and hits native Gemini API."""
        
        # Get the latest question from the user
        last_user_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
        user_query = last_user_message.content
        
        # Retrieve context matching the user's latest input
        retrieved_context = self.retrieve_knowledge(user_query)

        # ── Build the Native Gemini Contents list ───────────────────────────
        contents = []

        # Append full conversation history (all turns except the current one)
        for msg in state["messages"][:-1]:
            if isinstance(msg, HumanMessage):
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.content)]))
            elif isinstance(msg, AIMessage):
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg.content)]))

        # Inject RAG context right before the current question as user context
        if retrieved_context:
            context_text = (
                "السياق التعليمي التالي مأخوذ من قاعدة بيانات المنهج — استخدميه بدقة في شرحك:\n\n"
                f"{retrieved_context}"
            )
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=context_text)]))

        # Add the current user question
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_query)]))

        # Configure system instructions and parameters
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.7
        )

        try:
            print(f"LOG: Calling Native Gemini API ({GEMINI_MODEL}) with conversation history...")
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            ai_reply = response.text
        except Exception as e:
            ai_reply = f"خطأ في الاتصال بـ Gemini API: {e}"

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
    print(f"  LLM : {GEMINI_MODEL} (Native Google API)")
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

        # ── 🛡️ RUN LLM-AS-A-JUDGE EVALUATION ──────────────────────────────────
        print("LOG: Evaluating response with Llama-3.3-70b-Instruct...")
        
        # Pull matching context reference information used for this round
        rag_context_extracted = ""
        if tutor.collection.count() > 0:
            rag_context_extracted = tutor.retrieve_knowledge(user_input)
        
        # Trigger judge file validation engine
        evaluation = evaluate_response(
            user_query=user_input,
            rag_context=rag_context_extracted,
            assistant_response=final_response
        )
        
        # Print the finalized report structure
        print("═" * 50)
        print("🛡️  [JUDGE EVALUATION REPORT]")
        print("═" * 50)
        print(json.dumps(evaluation, indent=2, ensure_ascii=False))
        print("═" * 50 + "\n")