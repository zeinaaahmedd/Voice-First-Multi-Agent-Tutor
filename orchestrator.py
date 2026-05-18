"""
orchestrator.py  –  Master Orchestrator for the Voice-First Multi-Agent Tutor

Full pipeline per turn
    audio_path ──▶ STT ──▶ IntentClassifier ──▶ ┬── RAG (rag_db/) ──▶ PedagogicalAgent
                                                  │
                                                  └── lecture_sample.txt ──▶ SumQuizAgent
                                                                                  │
                                                                                  ▼
                                                                         TTS (Habibi) ──▶ .wav

Critical data-routing rules
    factual question  →  query rag_db/ for context  →  PedagogicalAgent
    summarize / quiz  →  read lecture_sample.txt     →  SumQuizAgent  (RAG bypassed)

Entry points
    TutorOrchestrator.process_audio(audio_path, ...)  – full voice pipeline
    TutorOrchestrator.process_text(text, ...)         – text-in (bypasses STT)
    python orchestrator.py [--no-tts] [--lecture path] [--audio path]
"""

from __future__ import annotations

import importlib.util
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from stt_agent import WhisperXSTT

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")

# Canonical project root (the directory this file lives in)
PROJECT_ROOT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# Intent Classifier  –  keyword-based Arabic router
# ──────────────────────────────────────────────────────────────────────────────
class IntentClassifier:
    """
    Maps an Arabic utterance to one of three intents:
        "pedagogical"  –  default; routes to the Pedagogical Agent
        "summarize"    –  routes to the Summarization Agent
        "quiz"         –  routes to the Quiz Agent
    """

    _QUIZ_KEYWORDS = {
        "كويز", "quiz", "اختبار", "امتحان", "امتحنني", "اختبرني",
        "اسألني", "أسئلة", "اعمل كويز", "اعمل اختبار", "ابدأ اختبار",
        "حل أسئلة", "أسئلة متعددة",
    }
    _SUMMARIZE_KEYWORDS = {
        "لخص", "ملخص", "خلاصة", "تلخيص", "summarize", "summary",
        "اشرح المحاضرة", "عاوز ملخص", "ابي ملخص", "اعطني ملخص",
        "وضح المحاضرة", "ما ملخص",
    }

    def classify(self, text: str) -> str:
        lower = text.lower()
        for kw in self._QUIZ_KEYWORDS:
            if kw in lower:
                log.info("Intent → quiz  (matched: '%s')", kw)
                return "quiz"
        for kw in self._SUMMARIZE_KEYWORDS:
            if kw in lower:
                log.info("Intent → summarize  (matched: '%s')", kw)
                return "summarize"
        log.info("Intent → pedagogical  (default)")
        return "pedagogical"


# ──────────────────────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TutorSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chat_history: list = field(default_factory=list)
    current_mode: str = "idle"      # idle | teaching | summarizing | quizzing
    pending_lecture: str = ""       # last lecture text supplied for sum/quiz
    last_response: str = ""
    last_audio_output: Optional[str] = None
    turn_count: int = 0

    def push(self, role: str, text: str):
        self.chat_history.append(
            {"role": role, "text": text, "turn": self.turn_count}
        )

    def recent_history(self, n: int = 6) -> str:
        lines = [
            f"Session {self.session_id[:8]} | "
            f"mode={self.current_mode} | turns={self.turn_count}"
        ]
        for e in self.chat_history[-n:]:
            lines.append(f"  [{e['role']:9s}] {e['text'][:80]}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Agent Adapters
# ──────────────────────────────────────────────────────────────────────────────
class PedagogicalAdapter:
    """Thin wrapper around PedagogicalAgentGraph."""

    def __init__(self):
        from pedagogical_agent import PedagogicalAgentGraph
        self._agent = PedagogicalAgentGraph()
        log.info("PedagogicalAdapter ready")

    def retrieve_context(self, query: str) -> str:
        """
        Query rag_db/ for context relevant to *query*.
        Delegates to the agent's own embedder + ChromaDB — no model duplication.
        Called by the orchestrator before routing to confirm relevant content exists.
        """
        return self._agent.retrieve_knowledge(query)

    def ask(self, question: str, thread_id: str) -> str:
        from langchain_core.messages import HumanMessage
        config = {"configurable": {"thread_id": thread_id}}
        state = self._agent.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        messages = state.get("messages", [])
        return messages[-1].content if messages else "عذراً، لم أتمكن من الإجابة."


class SumQuizAdapter:
    """
    Thin wrapper around SummaryQuizAgent.

    SummaryQuizAgent loads its prompt templates using bare relative paths
    (e.g. "chunk_summary_prompt.txt"), so we temporarily change CWD to
    PROJECT_ROOT before constructing or running it.
    The module's own file name contains spaces, so we load it via
    importlib.util.spec_from_file_location.
    """

    def __init__(self):
        spec = importlib.util.spec_from_file_location(
            "summarization_and_quiz_agent",
            PROJECT_ROOT / "summarization and quiz agent.py",
        )
        mod = importlib.util.module_from_spec(spec)
        self._run_in_project_root(spec.loader.exec_module, mod)
        self._cls = mod.SummaryQuizAgent
        log.info("SumQuizAdapter ready")

    @staticmethod
    def _run_in_project_root(fn, *args, **kwargs):
        prev = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            return fn(*args, **kwargs)
        finally:
            os.chdir(prev)

    def _build_and_run(self, text: str) -> dict:
        prev = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            agent = self._cls()
            return agent.run(text)
        finally:
            os.chdir(prev)

    def summarize(self, text: str) -> str:
        return self._build_and_run(text).get("summary", "لم أتمكن من إنشاء الملخص.")

    def quiz(self, text: str) -> str:
        return self._build_and_run(text).get("quiz", "لم أتمكن من إنشاء الكويز.")


class TTSAdapter:
    """
    Thin wrapper around run_habibi_tts_agent.

    ref_audio_path and ref_text both default to the values baked into
    HabibiTTSConfig (ref_speaker.wav + the Egyptian-Arabic reference text).
    Pass explicit values to override; pass empty strings to disable TTS.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        ref_audio_path: str = "",
        ref_text: str = "",
    ):
        from habibi_tts_service import _DEFAULT_REF_AUDIO, _DEFAULT_REF_TEXT

        self.output_dir = PROJECT_ROOT / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Fall back to the built-in defaults when caller passes nothing
        self.ref_audio_path = ref_audio_path or _DEFAULT_REF_AUDIO
        self.ref_text = ref_text or _DEFAULT_REF_TEXT

        ref_wav = Path(self.ref_audio_path)
        if not ref_wav.exists():
            log.warning(
                "TTS disabled: reference WAV not found at '%s'. "
                "Run the conversion step or pass --ref-audio.",
                self.ref_audio_path,
            )
            self._enabled = False
        else:
            self._enabled = True
            log.info("TTS enabled  (ref=%s)", ref_wav.name)

    def speak(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        if not self._enabled:
            return None

        from habibi_tts_graph import run_habibi_tts_agent
        from habibi_tts_service import HabibiTTSConfig

        fname = filename or f"response_{uuid.uuid4().hex[:8]}.wav"
        out_path = str(self.output_dir / fname)

        config = HabibiTTSConfig(
            ref_audio_path=self.ref_audio_path,
            ref_text=self.ref_text,
        )
        result = run_habibi_tts_agent(text=text, output_path=out_path, config=config)

        if not result.get("success", False):
            log.error("TTS synthesis failed: %s", result.get("error", "unknown"))
            return None

        return result.get("output_path", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Master Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
class TutorOrchestrator:
    """
    Central state machine for the Voice-First Multi-Agent Tutor.

    All agent adapters are lazily loaded on first use so that startup is
    fast and only the relevant models are loaded per session.

    Args:
        enable_tts:      Synthesise audio output for every response.
        stt_model_size:  whisper/whisperx model size ("tiny", "base", "small", …).
        tts_output_dir:  Directory where output WAV files are written.
        ref_audio_path:  Reference speaker WAV for Habibi TTS (voice cloning).
        ref_text:        Transcript of the reference audio.
    """

    def __init__(
        self,
        enable_tts: bool = True,
        stt_model_size: str = "base",
        tts_output_dir: str = "outputs",
        ref_audio_path: str = "",
        ref_text: str = "",
        lecture_path: Optional[str] = None,
    ):
        self.sessions: dict[str, TutorSession] = {}
        self.stt = WhisperXSTT(language="ar", model_size=stt_model_size)
        self.intent = IntentClassifier()

        # Canonical lecture file — used as the data source for summarize / quiz.
        # RAG is bypassed entirely for those intents.
        self._lecture_path: Path = (
            Path(lecture_path) if lecture_path else PROJECT_ROOT / "lecture_sample.txt"
        )

        self._pedagogical: Optional[PedagogicalAdapter] = None
        self._sumquiz: Optional[SumQuizAdapter] = None
        self._tts: Optional[TTSAdapter] = (
            TTSAdapter(
                output_dir=tts_output_dir,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
            )
            if enable_tts
            else None
        )
        log.info("TutorOrchestrator ready  (tts=%s)", enable_tts)

    # ── session management ───────────────────────────────────────────────────

    def get_or_create_session(self, session_id: Optional[str] = None) -> TutorSession:
        sid = session_id or str(uuid.uuid4())
        if sid not in self.sessions:
            self.sessions[sid] = TutorSession(session_id=sid)
            log.info("New session created: %s", sid)
        return self.sessions[sid]

    # ── lazy adapters ────────────────────────────────────────────────────────

    @property
    def pedagogical(self) -> PedagogicalAdapter:
        if self._pedagogical is None:
            self._pedagogical = PedagogicalAdapter()
        return self._pedagogical

    @property
    def sumquiz(self) -> SumQuizAdapter:
        if self._sumquiz is None:
            self._sumquiz = SumQuizAdapter()
        return self._sumquiz

    # ── public API ───────────────────────────────────────────────────────────

    def process_audio(
        self,
        audio_path: str,
        session_id: Optional[str] = None,
        lecture_text: Optional[str] = None,
    ) -> dict:
        """
        Full voice pipeline: audio file → text → agent → TTS wav.

        Args:
            audio_path:    Path to input WAV / MP3 file.
            session_id:    Resume an existing session (None = new session).
            lecture_text:  Lecture body to use for summarize / quiz intents.

        Returns::
            {
                "session_id":    str,
                "transcription": str,
                "intent":        str,        # "pedagogical" | "summarize" | "quiz"
                "response_text": str,
                "audio_output":  str | None, # path to output WAV, or None
            }
        """
        session = self.get_or_create_session(session_id)
        session.turn_count += 1
        log.info("[turn %d] STT ← %s", session.turn_count, audio_path)
        transcription = self.stt.transcribe(audio_path)
        log.info("[turn %d] STT → '%s'", session.turn_count, transcription[:80])
        session.push("user", transcription)
        return self._dispatch(session, transcription, lecture_text)

    def process_text(
        self,
        text: str,
        session_id: Optional[str] = None,
        lecture_text: Optional[str] = None,
    ) -> dict:
        """
        Text-in pipeline (bypasses STT).  Same return shape as process_audio().
        Ideal for testing and for integrations that already have a transcription.
        """
        session = self.get_or_create_session(session_id)
        session.turn_count += 1
        log.info("[turn %d] text-in '%s'", session.turn_count, text[:80])
        session.push("user", text)
        return self._dispatch(session, text, lecture_text)

    def session_info(self, session_id: str) -> str:
        session = self.sessions.get(session_id)
        return session.recent_history() if session else "Session not found."

    # ── internal dispatch ────────────────────────────────────────────────────

    def _load_lecture(self, override: Optional[str] = None) -> str:
        """
        Return the lecture text for summarize / quiz intents.
        Priority: explicit override → lecture_sample.txt  (RAG is never used here).
        """
        if override:
            return override
        if not self._lecture_path.exists():
            log.error("Lecture file not found: %s", self._lecture_path)
            return ""
        text = self._lecture_path.read_text(encoding="utf-8")
        log.info("Lecture loaded from %s  (%d chars)", self._lecture_path.name, len(text))
        return text

    def _dispatch(
        self,
        session: TutorSession,
        user_text: str,
        lecture_text: Optional[str],
    ) -> dict:
        intent = self.intent.classify(user_text)
        session.current_mode = {
            "pedagogical": "teaching",
            "summarize": "summarizing",
            "quiz": "quizzing",
        }[intent]

        rag_context: str = ""

        try:
            if intent == "pedagogical":
                # ── RAG path ────────────────────────────────────────────────
                # The orchestrator queries rag_db/ first so it owns the routing
                # decision. The pedagogical agent then re-uses the same loaded
                # embedder + ChromaDB internally when it builds its LLM prompt.
                rag_context = self.pedagogical.retrieve_context(user_text)
                log.info(
                    "[RAG] %d chars retrieved for query '%s'",
                    len(rag_context), user_text[:60],
                )
                response_text = self._route_pedagogical(session, user_text)

            elif intent == "summarize":
                # ── lecture_sample.txt path — RAG bypassed ──────────────────
                response_text = self._route_summarize(session, lecture_text)

            else:
                # ── lecture_sample.txt path — RAG bypassed ──────────────────
                response_text = self._route_quiz(session, lecture_text)

        except Exception:
            log.exception("Agent error  (intent=%s)", intent)
            response_text = "حصل خطأ في المعالجة. حاول تاني من فضلك."

        session.push("assistant", response_text)
        session.last_response = response_text

        audio_output: Optional[str] = None
        if self._tts:
            audio_output = self._tts.speak(response_text)
            session.last_audio_output = audio_output

        return {
            "session_id": session.session_id,
            "transcription": user_text,
            "intent": intent,
            "rag_context": rag_context,      # non-empty only for pedagogical turns
            "response_text": response_text,
            "audio_output": audio_output,
        }

    # ── per-intent handlers ──────────────────────────────────────────────────

    def _route_pedagogical(self, session: TutorSession, question: str) -> str:
        # RAG was already queried in _dispatch; the agent re-runs it internally
        # to inject the retrieved context directly into its LLM prompt.
        return self.pedagogical.ask(question, thread_id=session.session_id)

    def _route_summarize(self, session: TutorSession, lecture_text: Optional[str]) -> str:
        # RAG is bypassed. Source is always lecture_sample.txt (or explicit override).
        text = self._load_lecture(lecture_text) or session.pending_lecture
        if not text:
            return "مش لاقي نص المحاضرة. تأكد إن lecture_sample.txt موجود في المجلد."
        session.pending_lecture = text
        return self.sumquiz.summarize(text)

    def _route_quiz(self, session: TutorSession, lecture_text: Optional[str]) -> str:
        # RAG is bypassed. Source is always lecture_sample.txt (or explicit override).
        text = self._load_lecture(lecture_text) or session.pending_lecture
        if not text:
            return "مش لاقي نص المحاضرة. تأكد إن lecture_sample.txt موجود في المجلد."
        session.pending_lecture = text
        return self.sumquiz.quiz(text)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Voice-First Multi-Agent Tutor — Orchestrator"
    )
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    parser.add_argument(
        "--session", default=None, metavar="ID",
        help="Resume an existing session by its UUID",
    )
    parser.add_argument(
        "--lecture", default=None, metavar="PATH",
        help=(
            "Override the lecture file for summarize / quiz. "
            "Defaults to lecture_sample.txt in the project root. "
            "RAG is never used for summarize / quiz regardless of this flag."
        ),
    )
    parser.add_argument(
        "--audio", default=None, metavar="PATH",
        help="Process a single audio file then exit",
    )
    parser.add_argument(
        "--ref-audio", default="", metavar="PATH",
        help="Reference speaker WAV for Habibi TTS (required for --no-tts to be False)",
    )
    parser.add_argument(
        "--ref-text", default="", metavar="TEXT",
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--stt-model", default="base", metavar="SIZE",
        help="Whisper model size: tiny | base | small | medium | large  (default: base)",
    )
    args = parser.parse_args()

    lecture_text: Optional[str] = None
    if args.lecture:
        lecture_text = Path(args.lecture).read_text(encoding="utf-8")
        log.info("Lecture loaded  (%d chars)", len(lecture_text))

    orchestrator = TutorOrchestrator(
        enable_tts=not args.no_tts,
        stt_model_size=args.stt_model,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        lecture_path=args.lecture,   # None → auto-uses lecture_sample.txt
    )
    session_id = args.session

    # ── single-shot audio mode ───────────────────────────────────────────────
    if args.audio:
        result = orchestrator.process_audio(
            audio_path=args.audio,
            session_id=session_id,
            lecture_text=lecture_text,
        )
        _print_result(result)
        return

    # ── interactive text loop ────────────────────────────────────────────────
    _print_banner()
    while True:
        try:
            user_input = input("أنت: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nمع السلامة!")
            break

        if user_input.lower() in {"exit", "quit", "خروج"}:
            print("مع السلامة!")
            break
        if not user_input:
            continue

        result = orchestrator.process_text(
            text=user_input,
            session_id=session_id,
            lecture_text=lecture_text,
        )
        session_id = result["session_id"]
        _print_result(result)


def _print_banner():
    print("\n" + "═" * 62)
    print("  مرحبا! أنا بصيرة — المساعد التعليمي الصوتي")
    print("  اكتب سؤالك بالعربي، أو جرّب:")
    print("    'لخص'     ← ملخص المحاضرة  (يحتاج --lecture)")
    print("    'كويز'    ← أسئلة على المحاضرة  (يحتاج --lecture)")
    print("    'exit'    ← خروج")
    print("═" * 62 + "\n")


def _print_result(result: dict):
    print(f"\n  [Intent: {result['intent']}]")
    if result.get("transcription") and result["transcription"] != result.get("response_text"):
        print(f"  [STT]:  {result['transcription']}")
    if result.get("rag_context"):
        preview = result["rag_context"][:120].replace("\n", " ")
        print(f"  [RAG]:  {preview}…")
    elif result["intent"] in ("summarize", "quiz"):
        print("  [RAG]:  bypassed — using lecture_sample.txt")
    print(f"\nبصيرة: {result['response_text']}\n")
    if result.get("audio_output"):
        print(f"  [Audio saved → {result['audio_output']}]\n")


if __name__ == "__main__":
    _cli()
