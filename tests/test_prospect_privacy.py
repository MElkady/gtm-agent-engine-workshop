import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import gtm_agent


class ProspectPrivacyTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_allowlist(self):
        record = {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "disqualified": False,
            "annual_revenue": 75000000,
            "enrichment_source": "crm",
            "billing_qualification": {"tax_id": "secret"},
            "extra_field": "not allowed",
        }
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(result["prospect"], {
            "prospect_id": "LEAD-1",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "disqualified": False,
            "annual_revenue": 75000000,
            "enrichment_source": "crm",
        })

    def test_build_profile_excludes_sensitive_data_from_result_and_cache(self):
        record = {
            "name": "Ada Lovelace",
            "annual_revenue": 75000000,
            "billing_qualification": {"tax_id": "secret"},
        }
        save_profile = Mock()
        with (
            patch.object(gtm_agent.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(gtm_agent.data_service, "get_prospect_record", return_value=record),
            patch.object(gtm_agent.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(gtm_agent.data_service, "fetch_account_details", return_value={}),
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=["Snowflake"]),
            patch.object(gtm_agent.data_service, "save_profile_to_db", save_profile),
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        profile = result["prospect_profile"]
        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("billing_qualification", save_profile.call_args.args[1])
        self.assertEqual(profile["annual_revenue"], 75000000)
        self.assertEqual(profile["tech_stack"], ["Snowflake"])

    def test_build_profile_strips_sensitive_data_from_cached_profile(self):
        cached = {
            "prospect_id": "LEAD-1",
            "annual_revenue": 75000000,
            "tech_stack": ["Snowflake"],
            "billing_qualification": {"tax_id": "secret"},
        }
        with patch.object(
            gtm_agent.data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": cached},
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertEqual(result["prospect_profile"]["annual_revenue"], 75000000)

    def test_score_prospect_omits_sensitive_data_from_scoring_prompt(self):
        profile = {
            "prospect_id": "LEAD-1",
            "annual_revenue": 75000000,
            "tech_stack": ["Snowflake"],
            "billing_qualification": {"tax_id": "secret"},
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 50000000,
            "description": "Platform",
        }
        response = Mock()
        response.model_dump.return_value = {"score": 90}
        llm = Mock()
        llm.invoke.return_value = response
        with (
            patch.object(gtm_agent, "_scoring_llm", llm),
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=["Snowflake"]),
        ):
            result = gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        prompt = llm.invoke.call_args.args[0][1]["content"]
        serialized_profile = json.loads(prompt.split("\n\nProspect profile:\n", 1)[1])
        self.assertNotIn("billing_qualification", serialized_profile)
        self.assertNotIn("secret", prompt)
        self.assertEqual(serialized_profile["annual_revenue"], 75000000)
        self.assertEqual(serialized_profile["tech_stack"], ["Snowflake"])
        self.assertEqual(result, {"score": 90})


if __name__ == "__main__":
    unittest.main()
