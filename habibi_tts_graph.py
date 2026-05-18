from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from habibi_tts_service import HabibiTTSService, HabibiTTSConfig


class HabibiTTSState(TypedDict, total=False):
    text: str
    output_path: str | None
    result: dict
    error: str


def validate_input(state: HabibiTTSState) -> HabibiTTSState:
    text = (state.get("text") or "").strip()
    if not text:
        return {
            "error": "Empty input",
            "result": {
                "success": False,
                "output_path": None,
                "sample_rate": None,
                "text": state.get("text", ""),
                "chunks_count": 0,
                "error": "Empty input",
            },
        }
    return {"text": text, "output_path": state.get("output_path")}


def run_tts(service: HabibiTTSService, state: HabibiTTSState) -> HabibiTTSState:
    if state.get("error"):
        return {}
    result = service.synthesize(state["text"], output_path=state.get("output_path"))
    return {"result": result}


def finalize_result(state: HabibiTTSState) -> HabibiTTSState:
    if state.get("result"):
        return {}
    if state.get("error"):
        return {
            "result": {
                "success": False,
                "output_path": None,
                "sample_rate": None,
                "text": state.get("text", ""),
                "chunks_count": 0,
                "error": state["error"],
            }
        }
    return {}


def build_habibi_tts_graph(service: HabibiTTSService):
    builder = StateGraph(HabibiTTSState)
    builder.add_node("validate_input", validate_input)
    builder.add_node("run_tts", lambda s: run_tts(service, s))
    builder.add_node("finalize_result", finalize_result)

    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "run_tts")
    builder.add_edge("run_tts", "finalize_result")
    builder.add_edge("finalize_result", END)

    return builder.compile()


def run_habibi_tts_agent(
    text: str,
    output_path: str | None = None,
    service: HabibiTTSService | None = None,
    config: HabibiTTSConfig | None = None,
) -> dict:
    if service is None:
        service = HabibiTTSService(config or HabibiTTSConfig())

    graph = build_habibi_tts_graph(service)
    state = graph.invoke({"text": text, "output_path": output_path})
    return state.get("result", {})
