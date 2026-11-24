import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import feed

GEMINI_TEXT = """```json
[{"title":"Director, Marketing Operations","company":"Braze, Inc.","url":"https://job-boards.greenhouse.io/braze/jobs/123?gh_src=abc&utm_source=x","location":"New York, NY","posted_date":"2026-11-20","comp":"not disclosed","why_fit":"AEP/AJO depth"},
 {"title":"VP Partner Marketing","company":"Current Employer Inc","url":"https://example.com/jobs/9","location":"NYC","posted_date":"unknown","comp":"","why_fit":"x"}]
```"""
GROK_TEXT = """Here you go:
[{"title":"Sr. Director Marketing Operations","company":"Braze","url":"https://www.linkedin.com/jobs/view/555","location":"NYC","posted_date":"unknown","comp":"$220K-$260K","why_fit":"ops leader"},
 {"title":"Head of Demand Generation","company":"Klaviyo","url":"https://job-boards.greenhouse.io/klaviyo/jobs/77","location":"Remote","posted_date":"2026-11-18","comp":"not disclosed","why_fit":"demand gen"}]"""

def test_pipeline():
    spec = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "example_client.json")))
    feed.MODELS = {"gemini": lambda p: GEMINI_TEXT, "grok": lambda p: GROK_TEXT}
    feed.check_link = lambda r: r.update(link_status="live", link_note="", is_aggregator_url=feed.is_aggregator(r["url"]))
    out = feed.run_client(spec)
    roles = out["roles"]
    assert out["counts"]["raw_per_model"] == {"gemini": 2, "grok": 2}
    assert not any(r["company"] == "Current Employer Inc" for r in roles), "excluded company leaked"
    braze = [r for r in roles if "braze" in r["company"].lower()]
    assert len(braze) == 1, "Braze duplicate not merged"
    b = braze[0]
    assert b["consensus_count"] == 2 and b["source_models"] == ["gemini", "grok"]
    assert b["comp"] == "$220K-$260K", "comp not backfilled from second model"
    assert "gh_src" not in b["url_norm"] and "utm" not in b["url_norm"]
    assert roles[0] is b, "multi-model role should rank first"
    assert len(roles) == 2
    assert "Current Employer Inc" in spec["excluded_companies"]
    assert "Today is" in feed.build_prompt(spec)
    print(json.dumps(out["counts"]), "OK")

if __name__ == "__main__":
    test_pipeline()
