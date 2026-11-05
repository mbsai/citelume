import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import source_budget as sb


class TestOpenAlexBudget(unittest.TestCase):
    def setUp(self):
        sb.OPENALEX_CREDITS.reset()
        sb.begin_search_budget()

    def test_server_key_is_capped_daily_and_per_search(self):
        env = {"DEMO_OPENALEX_API_KEY": "srv", "OPENALEX_DAILY_CREDIT_CAP": "30",
               "OPENALEX_MAX_CALLS_PER_SEARCH": "5", "DEMO_STATE_DIR": ""}
        with mock.patch.dict(os.environ, env):
            self.assertEqual([sb.openalex_call_allowed("srv") for _ in range(3)], [None, None, None])
            self.assertIn("daily", sb.openalex_call_allowed("srv"))  # 3 searches x 10 credits = cap
            self.assertEqual(sb.OPENALEX_CREDITS.used_today(), 30)
        sb.OPENALEX_CREDITS.reset()
        sb.begin_search_budget()
        with mock.patch.dict(os.environ, {**env, "OPENALEX_DAILY_CREDIT_CAP": "1000", "OPENALEX_MAX_CALLS_PER_SEARCH": "2"}):
            sb.openalex_call_allowed("srv"), sb.openalex_call_allowed("srv")
            self.assertIn("this search", sb.openalex_call_allowed("srv"))

    def test_visitor_keys_and_keyless_calls_are_not_counted(self):
        with mock.patch.dict(os.environ, {"DEMO_OPENALEX_API_KEY": "srv", "OPENALEX_DAILY_CREDIT_CAP": "0"}):
            self.assertIsNone(sb.openalex_call_allowed("visitor-key"))
            self.assertIsNone(sb.openalex_call_allowed(""))
            self.assertEqual(sb.OPENALEX_CREDITS.used_today(), 0)


if __name__ == "__main__":
    unittest.main()
