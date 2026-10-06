import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanoclaw.core import provider


USER_ERROR = (
    "Error code: 400 - {'error': {'message': "
    "'invalid temperature: only 1 is allowed for this model', "
    "'type': 'invalid_request_error'}}"
)


class TestTemperatureDetection(unittest.TestCase):
    def test_reasoning_model_names_lock_to_one(self):
        for name in ("o3", "o3-mini", "o4-mini", "o1-preview", "openai/gpt-5", "gpt-5-nano"):
            self.assertTrue(provider.model_locks_temperature_to_one(name), name)

    def test_normal_models_keep_requested_temperature(self):
        for name in ("gpt-4o-mini", "qwen-max", "glm-4", "gpt-5-chat"):
            self.assertFalse(provider.model_locks_temperature_to_one(name), name)

    def test_user_error_requires_temperature_one(self):
        self.assertEqual(provider.temperature_required_by_error(RuntimeError(USER_ERROR)), 1.0)

    def test_unrelated_error_is_ignored(self):
        self.assertIsNone(provider.temperature_required_by_error(RuntimeError("invalid api key")))


class TestTemperatureRetry(unittest.TestCase):
    def setUp(self):
        provider._FORCED_TEMPERATURE.clear()

    def test_rejected_temperature_is_retried_as_one(self):
        from langchain_openai import ChatOpenAI

        seen = []

        def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(kwargs.get("temperature", self.temperature))
            if len(seen) == 1:
                raise RuntimeError(USER_ERROR)
            return "ok"

        with patch.object(ChatOpenAI, "_generate", fake_generate):
            llm = provider.get_provider(
                provider_name="other",
                model_name="custom-reasoner",
                api_key="sk-test",
                base_url="http://127.0.0.1:9/v1",
            )
            self.assertEqual(llm._generate([]), "ok")

        self.assertEqual(seen, [0.0, 1.0])
        self.assertEqual(provider._FORCED_TEMPERATURE["custom-reasoner"], 1.0)

    def test_known_reasoning_model_starts_at_one(self):
        llm = provider.get_provider(
            provider_name="other",
            model_name="o3-mini",
            api_key="sk-test",
            base_url="http://127.0.0.1:9/v1",
        )
        self.assertEqual(llm.temperature, 1.0)

    def test_other_errors_are_not_retried(self):
        from langchain_openai import ChatOpenAI

        calls = {"n": 0}

        def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
            calls["n"] += 1
            raise RuntimeError("connection refused")

        with patch.object(ChatOpenAI, "_generate", fake_generate):
            llm = provider.get_provider(
                provider_name="other",
                model_name="plain-model",
                api_key="sk-test",
                base_url="http://127.0.0.1:9/v1",
            )
            with self.assertRaisesRegex(RuntimeError, "connection refused"):
                llm._generate([])

        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
