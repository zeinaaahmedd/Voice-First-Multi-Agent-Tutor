import unittest

from habibi_tts_graph import run_habibi_tts_agent


class FakeService:
    def synthesize(self, text: str, output_path: str | None = None) -> dict:
        return {
            "success": True,
            "output_path": output_path or "out.wav",
            "sample_rate": 24000,
            "text": text,
            "chunks_count": 1,
        }


class HabibiTTSSmokeTests(unittest.TestCase):
    def test_empty_input_validation(self):
        result = run_habibi_tts_agent("", service=FakeService())
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Empty input")

    def test_agent_invocation_shape(self):
        result = run_habibi_tts_agent("hello", service=FakeService())
        self.assertTrue(result["success"])
        self.assertIn("output_path", result)
        self.assertIn("sample_rate", result)
        self.assertIn("chunks_count", result)


if __name__ == "__main__":
    unittest.main()
