import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ApiKeysConfig
from services import demo_mode

ENV = {"CITELUME_DEMO": "1", "DEMO_OPENROUTER_API_KEY": "server-key", "DEMO_STATE_DIR": ""}


class TestDemoMode(unittest.TestCase):
    def setUp(self):
        demo_mode.USAGE.reset()

    def test_no_key_uses_server_key_and_nvidia_model(self):
        with mock.patch.dict(os.environ, ENV):
            cfg, server = demo_mode.apply_demo_config(ApiKeysConfig(llm_provider="gemini", llm_model="gemini-2.5-flash"))
        self.assertTrue(server)
        self.assertEqual(cfg.llm_provider, "openrouter")
        self.assertTrue(cfg.llm_model.startswith("nvidia/") and cfg.llm_model.endswith(":free"))
        self.assertTrue(all(m.startswith("nvidia/") for m in cfg.llm_fallback_models))

    def test_own_key_is_kept_and_not_on_demo_quota(self):
        with mock.patch.dict(os.environ, ENV):
            cfg, server = demo_mode.apply_demo_config(ApiKeysConfig(llm_provider="openai", llm_api_key="mine", llm_model="gpt-x"))
        self.assertFalse(server)
        self.assertEqual((cfg.llm_api_key, cfg.llm_model), ("mine", "gpt-x"))
        self.assertEqual(demo_mode.bucket_for("deep_research", server), "own_key")

    def test_local_servers_blocked(self):
        with mock.patch.dict(os.environ, ENV):
            for cfg in (ApiKeysConfig(llm_provider="ollama"),
                        ApiKeysConfig(llm_provider="custom", llm_api_key="k", llm_base_url="http://127.0.0.1:8080/v1"),
                        ApiKeysConfig(llm_provider="custom", llm_api_key="k", llm_base_url="http://10.0.0.5/v1")):
                with self.assertRaises(demo_mode.DemoLimitError):
                    demo_mode.apply_demo_config(cfg)

    def test_daily_limits_per_address(self):
        with mock.patch.dict(os.environ, ENV):
            for _ in range(3):
                demo_mode.take_quota("1.2.3.4", "standard")
            with self.assertRaises(demo_mode.DemoLimitError):
                demo_mode.take_quota("1.2.3.4", "standard")
            demo_mode.take_quota("1.2.3.4", "deep_research")
            with self.assertRaises(demo_mode.DemoLimitError):
                demo_mode.take_quota("1.2.3.4", "deep_research")
            demo_mode.take_quota("5.6.7.8", "standard")  # another address has its own allowance
            self.assertEqual(demo_mode.status_for("1.2.3.4")["remaining"], {"standard": 0, "deep_research": 0})

    def test_client_ip_ignores_spoofed_left_entries(self):
        with mock.patch.dict(os.environ, {"DEMO_PROXY_HOPS": "1"}):
            self.assertEqual(demo_mode.client_ip("9.9.9.9, 203.0.113.7", "10.0.0.1"), "203.0.113.7")
            self.assertEqual(demo_mode.client_ip(None, "10.0.0.1"), "10.0.0.1")


if __name__ == "__main__":
    unittest.main()
