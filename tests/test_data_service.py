import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent
from gtm_agent.gtm_records import PROSPECTS


class DataServiceUpdateProspectInfoTests(unittest.TestCase):
    prospect_id = "LEAD-71001"

    def setUp(self):
        self.original_tech_stack = list(PROSPECTS[self.prospect_id]["tech_stack"])
        self.original_profile = data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        PROSPECTS[self.prospect_id]["tech_stack"] = self.original_tech_stack
        data_service._PROFILES.pop(self.prospect_id, None)
        if self.original_profile is not None:
            data_service._PROFILES[self.prospect_id] = self.original_profile

    def test_update_persists_tech_stack_for_fetch(self):
        data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))

    def test_update_refreshes_cached_profile(self):
        gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        profile = gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]

        self.assertIn("Kafka", profile["tech_stack"])

    def test_update_result_reports_stored_tech_stack(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertEqual(result["tech_stack"], PROSPECTS[self.prospect_id]["tech_stack"])
        self.assertIn("Kafka", result["tech_stack"])

    def test_unknown_prospect_does_not_change_profile_cache(self):
        unknown_id = "LEAD-UNKNOWN"
        self.assertNotIn(unknown_id, data_service._PROFILES)

        result = data_service.update_prospect_info(unknown_id, "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertNotIn(unknown_id, data_service._PROFILES)

        cached_profile = {"sentinel": True}
        data_service._PROFILES[unknown_id] = cached_profile
        self.addCleanup(data_service._PROFILES.pop, unknown_id, None)

        result = data_service.update_prospect_info(unknown_id, "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIs(data_service._PROFILES[unknown_id], cached_profile)
        self.assertEqual(cached_profile, {"sentinel": True})

    def test_scoring_receives_updated_saved_tech_stack(self):
        profile = gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]
        data_service.update_prospect_info(self.prospect_id, "Kafka")
        captured_profile = {}

        class ScoringResult:
            def model_dump(self):
                return {"score": 90, "justification": "Kafka is present."}

        class ScoringModel:
            def invoke(self, messages):
                profile_text = messages[1]["content"].split(
                    "\n\nProspect profile:\n", 1
                )[1]
                captured_profile.update(json.loads(profile_text))
                return ScoringResult()

        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 1,
            "description": "Test offering",
        }
        with patch.object(gtm_agent, "_scoring_llm", ScoringModel()):
            result = gtm_agent.score_prospect.invoke(
                {"prospect_profile": profile, "offering": offering}
            )

        self.assertIn("Kafka", captured_profile["tech_stack"])
        self.assertIn("Kafka", result["justification"])


if __name__ == "__main__":
    unittest.main()
