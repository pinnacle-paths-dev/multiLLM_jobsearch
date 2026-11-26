#!/usr/bin/env python3
"""
Pinnacle Paths — multi-LLM role feed.

For each client spec in config/, asks Gemini (Google Search grounding) and
Grok (xAI web_search tool) for open roles, normalizes the results to one
schema, removes duplicates, checks every link is live, scores consensus
(how many models found the same role), and emails a dated JSON feed as an
attachment via Resend. The Claude scheduled task finds it in Gmail and runs
the role-fit skill on it.

Environment variables (GitHub Secrets):
  GEMINI_API_KEY             Google AI Studio key
  XAI_API_KEY                xAI key
  RESEND_API_KEY             Resend key (sends the feed email)
  CLIENT_SPECS_JSON          JSON list of client specs
GitHub Actions variables:
  FEED_EMAIL_FROM            verified Resend sender, e.g. Role Feed <jobs@yourdomain>
  FEED_EMAIL_TO              inbox the Claude scheduled task reads
Optional:
  GEMINI_MODEL  (default gemini-3.5-flash)
  GROK_MODEL    (default grok-4.7)
  DRY_RUN=1     write feed to ./out/ instead of emailing
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import sys
import time
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests

GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"
GROK_MODEL = os.environ.get("GROK_MODEL") or "grok-4.7"
MUSE_MODEL = os.environ.get("MUSE_MODEL") or "muse-spark-1.3"
# Web-search calls can take several minutes; one generous attempt beats two short ones.
TIMEOUT = 420
UA = {"User-Agent": "Mozilla/5.0 (PinnaclePathsRoleFeed/1.0)"}

FIELDS = ["title", "company", "url", "location", "posted_date", "comp", "why_fit"]

TRACKING_PARAMS = re.compile(
    r"^(utm_.*|gh_src|gh_jid_src|source|src|ref|refId|trk|trackingId|lever-source|"
    r"lever-origin|mode|iis|iisn|ccuid|jobPipeline|codes|gclid|fbclid|mc_cid|mc_eid)$",
    re.I,
)
CLOSED_SIGNALS = [
    "no longer available", "no longer accepting", "position has been filled",
    "job not found", "job has expired", "listing has closed", "position filled",
    "not accepting applications", "job has been removed", "this job is closed",
    "page not found", "posting has been closed",
]
AGGREGATOR_HOSTS = (
    "linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com", "builtin",
    "wellfound.com", "simplyhired.com", "monster.com", "dice.com", "talent.com",
    "jooble", "adzuna", "efinancialcareers", "themuse.com", "lensa.com",
)
SENIORITY = {"senior", "sr", "director", "vp", "vice", "president", "svp", "evp",
             "head", "of", "the", "and", "&", "global", "lead", "principal", "-", ","}


# ---------------------------------------------------------------- prompt
def build_prompt(spec: dict) -> str:
    today = dt.date.today().isoformat()
    return f"""You are a senior executive recruiter. Today is {today}.
Use live web search to find CURRENTLY OPEN job postings for this candidate.

CANDIDATE: {spec['candidate_summary']}
TARGET FUNCTIONS: {', '.join(spec['target_functions'])}
SENIORITY: {', '.join(spec['target_seniority'])}
INDUSTRIES: {', '.join(spec['target_industries'])}
GEOGRAPHY: {spec['geography']}
COMPENSATION: {spec['comp_floor']}
EXCLUDE: {'; '.join(spec['exclusions'])}
EXCLUDED COMPANIES: {', '.join(spec.get('excluded_companies', [])) or 'none'}
EXAMPLE TITLES: {', '.join(spec['example_good_titles'])}

RULES
- Only roles posted within the last {spec['recency_days']} days and still open.
- Prefer the URL of the posting on the company's own careers site or ATS
  (Greenhouse, Lever, Ashby, Workday, SmartRecruiters).
- Copy every URL EXACTLY as it appears in your search results. Never build,
  guess, shorten or complete a URL or a job ID yourself. If you only found the
  role on an aggregator, give that aggregator URL; it will be checked.
- posted_date must come from the posting or search result itself; otherwise
  write "unknown". Never estimate it.
- Return at most {spec['max_roles_per_model']} roles, best fit first.

OUTPUT: only a JSON array, no prose, no markdown fences. Each item:
{{"title": str, "company": str, "url": str, "location": str,
  "posted_date": "YYYY-MM-DD or unknown", "comp": "range or not disclosed",
  "why_fit": "one line"}}"""


# ---------------------------------------------------------------- model calls
def call_gemini(prompt: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 1.0},
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def call_grok(prompt: str) -> str:
    key = os.environ["XAI_API_KEY"]
    r = requests.post(
        "https://api.x.ai/v1/responses",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": GROK_MODEL,
            "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search"}],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return responses_text(r.json())


MODELS = {"gemini": call_gemini, "grok": call_grok}


# ---------------------------------------------------------------- Muse (Meta Model API)
def call_muse(prompt: str) -> str:
    """Meta's Muse Spark via the OpenAI-compatible Responses API, with web search."""
    key = os.environ["MUSE_API_KEY"]
    body = {"model": MUSE_MODEL, "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search"}]}
    r = requests.post("https://api.meta.ai/v1/responses",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json=body, timeout=TIMEOUT)
    if r.status_code == 400 and "tool" in r.text.lower():
        # Tool name differs on this account: retry with the preview tool name.
        body["tools"] = [{"type": "web_search_preview"}]
        r = requests.post("https://api.meta.ai/v1/responses",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json=body, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Meta Model API HTTP {r.status_code}: {r.text[:200]}")
    return responses_text(r.json())


def responses_text(data: dict) -> str:
    """Pulls the text out of an OpenAI-style Responses API reply."""
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    texts = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                texts.append(c["text"])
    return "\n".join(texts)


if os.environ.get("MUSE_API_KEY"):
    MODELS["muse"] = call_muse


SECRET_ENV = ("GEMINI_API_KEY", "XAI_API_KEY", "MUSE_API_KEY", "RESEND_API_KEY")


def redact(text: str) -> str:
    """Never let an API key reach an email, attachment or log."""
    for var in SECRET_ENV:
        val = os.environ.get(var)
        if val and len(val) > 6:
            text = text.replace(val, "[REDACTED]")
    return re.sub(r"(api_key|key|token)=[^&\s'\"]+", r"\1=[REDACTED]", text, flags=re.I)


# ---------------------------------------------------------------- parsing
def extract_json_array(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("url") and d.get("title")]


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(").,;")
    parts = urlsplit(url)
    if not parts.scheme:
        parts = urlsplit("https://" + url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not TRACKING_PARAMS.match(k)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", parts.netloc.lower().removeprefix("www."), path,
                       urlencode(query), ""))


def title_tokens(title: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return frozenset(w for w in words if w not in SENIORITY)


def company_key(company: str) -> str:
    c = re.sub(r"[^a-z0-9 ]", "", company.lower())
    c = re.sub(r"\b(inc|llc|ltd|corp|corporation|co|the|technologies|group)\b", "", c)
    return " ".join(c.split())


def same_role(a: dict, b: dict) -> bool:
    if a["url_norm"] == b["url_norm"]:
        return True
    if company_key(a["company"]) != company_key(b["company"]):
        return False
    ta, tb = title_tokens(a["title"]), title_tokens(b["title"])
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


def merge(results: dict[str, list[dict]]) -> list[dict]:
    merged: list[dict] = []
    for model, roles in results.items():
        for raw in roles:
            role = {f: str(raw.get(f, "") or "").strip() for f in FIELDS}
            role["url_norm"] = normalize_url(role["url"])
            match = next((m for m in merged if same_role(m, role)), None)
            if match:
                if model not in match["source_models"]:
                    match["source_models"].append(model)
                match["alt_urls"] = sorted(set(match["alt_urls"]) | {role["url"]} - {match["url"]})
                for f in ("comp", "posted_date", "location"):
                    if match[f] in ("", "unknown", "not disclosed") and role[f]:
                        match[f] = role[f]
            else:
                role["source_models"] = [model]
                role["alt_urls"] = []
                merged.append(role)
    for m in merged:
        m["consensus_count"] = len(m["source_models"])
    return merged


# ---------------------------------------------------------------- checks
def is_aggregator(url: str) -> bool:
    host = urlsplit(url).netloc.lower()
    return any(a in host for a in AGGREGATOR_HOSTS)


GREENHOUSE_JOB = re.compile(r"greenhouse\.io/(?:embed/job_app\?for=)?([A-Za-z0-9_-]+)/jobs/(\d+)")


def wrong_link(role: dict, note: str) -> None:
    """The role may be real but the model gave a bad address. Keep it as a lead
    (status 'unreachable') so the Claude run searches the company's own site."""
    role["link_status"] = "unreachable"
    role["link_wrong"] = True
    role["link_note"] = f"WRONG LINK ({note}) - role may still be real; find the posting on the company's own careers site"


def check_greenhouse(role: dict, slug: str, job_id: str) -> bool:
    """Checks a Greenhouse link against the board's public API. Returns True if handled."""
    try:
        r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs/{job_id}",
                         headers=UA, timeout=25)
    except requests.RequestException:
        return False
    if r.status_code == 404:
        wrong_link(role, f"Greenhouse has no job {job_id} on board '{slug}'")
        return True
    if r.status_code != 200:
        return False
    data = r.json()
    role["link_status"] = "live"
    role["ats_title"] = data.get("title", "")
    role["final_url"] = data.get("absolute_url", role["url"])
    if data.get("first_published"):
        role["first_published"] = data["first_published"][:10]
    key = title_tokens(role["title"])
    got = title_tokens(role["ats_title"])
    if key and got and len(key & got) / len(key | got) < 0.5:
        role["link_note"] = f"ATS title differs: '{role['ats_title']}' - Claude to check it is the same role"
    else:
        role["link_note"] = "confirmed on Greenhouse board API"
    return True


def check_link(role: dict) -> None:
    """Fills link_status: live | closed | dead | unreachable (incl. wrong links)."""
    role["is_aggregator_url"] = is_aggregator(role["url"])
    role["link_wrong"] = False
    m = GREENHOUSE_JOB.search(role["url"])
    if m and check_greenhouse(role, m.group(1), m.group(2)):
        return
    try:
        r = requests.get(role["url"], headers=UA, timeout=25, allow_redirects=True)
    except requests.RequestException as e:
        role["link_status"], role["link_note"] = "unreachable", type(e).__name__
        return
    role["final_url"] = r.url
    final = r.url.lower()
    if "error=true" in final or "error=" in urlsplit(final).query:
        wrong_link(role, "job board redirected with an error flag")
        return
    # A job-specific address that lands on a page without its job ID is a silent
    # redirect to the board's home page, not a live posting.
    ids = re.findall(r"\d{6,}", urlsplit(role["url"]).path)
    if ids and not any(i in final for i in ids) and not role["is_aggregator_url"]:
        wrong_link(role, "redirected away from the job's own page")
        return
    if r.status_code >= 400:
        role["link_status"], role["link_note"] = "dead", f"HTTP {r.status_code}"
        return
    body = r.text.lower()[:400_000]
    hit = next((s for s in CLOSED_SIGNALS if s in body), None)
    if hit:
        role["link_status"], role["link_note"] = "closed", f"page says '{hit}'"
        return
    key = [w for w in title_tokens(role["title"]) if len(w) > 3]
    if key and not any(w in body for w in key):
        # JS-rendered ATS pages often hide the title; don't drop, just flag.
        role["link_status"], role["link_note"] = "live", "title not found in raw HTML (JS page?) — Claude to verify"
        return
    role["link_status"], role["link_note"] = "live", ""


def excluded(role: dict, spec: dict) -> bool:
    ck = company_key(role["company"])
    return any(company_key(x) == ck for x in spec.get("excluded_companies", []))


# ---------------------------------------------------------------- delivery
def send_via_resend(spec: dict, filename: str, payload: dict) -> str:
    """Emails the feed as a JSON attachment. The client's Claude scheduled task
    finds it in Gmail by subject and reads the attachment."""
    import base64
    c = payload["counts"]
    subject = f"[Role Feed] {spec['client_id']} {dt.date.today().isoformat()}"
    rows = "".join(
        f"<tr><td style='font-family:Arial;padding:2px 8px'>{r['company']}</td>"
        f"<td style='font-family:Arial;padding:2px 8px'>{r['title']}</td>"
        f"<td style='font-family:Arial;padding:2px 8px'>{'+'.join(r['source_models'])}</td>"
        f"<td style='font-family:Arial;padding:2px 8px'>{'wrong link' if r.get('link_wrong') else r['link_status']}</td></tr>"
        for r in payload["roles"][:25]
    )
    errs = "".join(f"<p style='font-family:Arial;color:#b00'>{k} error: {v}</p>"
                   for k, v in payload["model_errors"].items())
    html = (
        f"<p style='font-family:Arial'>Automated multi-LLM role feed for <b>{spec['client_name']}</b>. "
        f"Discovery leads only; the Claude run verifies each before anything reaches the client.</p>"
        f"<p style='font-family:Arial'>Gemini {c['raw_per_model'].get('gemini', 0)} · Grok "
        f"{c['raw_per_model'].get('grok', 0)} · Muse {c['raw_per_model'].get('muse', '-')} · after dedupe {c['after_dedupe']} · live {c['live']} · "
        f"found by both {c['multi_model']}</p>{errs}"
        f"<table style='border-collapse:collapse;font-size:12px'>{rows}</table>"
    )
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
                 "Content-Type": "application/json"},
        json={
            "from": os.environ["FEED_EMAIL_FROM"],
            "to": [os.environ["FEED_EMAIL_TO"]],
            "subject": subject,
            "html": html,
            "attachments": [{
                "filename": filename,
                "content": base64.b64encode(json.dumps(payload, indent=2).encode()).decode(),
            }],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("id", "")


# ---------------------------------------------------------------- main
def run_client(spec: dict) -> dict:
    prompt = build_prompt(spec)
    results, errors = {}, {}

    def ask(name, fn):
        for attempt in (1, 2):
            try:
                text = fn(prompt)
                roles = extract_json_array(text)
                if not roles:
                    # Answered, but not as a JSON list of roles: say so instead of a silent 0.
                    snippet = " ".join((text or "").split())[:160] or "(empty reply)"
                    return [], redact(f"no roles in reply: {snippet}")
                return roles, None
            except Exception as e:  # noqa: BLE001
                err = redact(f"{type(e).__name__}: {e}")[:300]
                if isinstance(e, requests.Timeout) or attempt == 2:
                    return [], err  # a timeout already waited TIMEOUT seconds
                time.sleep(5)
        return [], "no attempt made"

    # The models run at the same time, so a slow one doesn't hold up the others.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {name: pool.submit(ask, name, fn) for name, fn in MODELS.items()}
    for name, fut in futures.items():
        results[name], err = fut.result()
        if err and not results[name]:
            errors[name] = err
        print(f"  {name}: {len(results[name])} roles", file=sys.stderr)

    roles = [r for r in merge(results) if not excluded(r, spec)]
    for r in roles:
        check_link(r)
    status_rank = {"live": 0, "unreachable": 1, "closed": 2, "dead": 3}
    roles.sort(key=lambda r: (status_rank[r["link_status"]], -r["consensus_count"],
                              r["is_aggregator_url"]))
    return {
        "client_id": spec["client_id"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "models": {"gemini": GEMINI_MODEL, "grok": GROK_MODEL,
                   **({"muse": MUSE_MODEL} if "muse" in MODELS else {})},
        "model_errors": errors,
        "counts": {
            "raw_per_model": {k: len(v) for k, v in results.items()},
            "after_dedupe": len(roles),
            "live": sum(r["link_status"] == "live" for r in roles),
            "multi_model": sum(r["consensus_count"] > 1 for r in roles),
        },
        "note": "Discovery only. Links here are NOT verified on the company site; the Claude run must verify each before it reaches any email.",
        "roles": roles,
    }


def load_specs() -> list[dict]:
    """Client specs come from the CLIENT_SPECS_JSON secret (a JSON list) so no
    client data lives in the repo. Falls back to config/*.json for local runs,
    skipping the example file."""
    raw = os.environ.get("CLIENT_SPECS_JSON", "").strip()
    if raw:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "config", "*.json")))
    return [json.load(open(p)) for p in paths if not p.endswith("example_client.json")]


def main() -> int:
    dry = os.environ.get("DRY_RUN") == "1"
    only = sys.argv[1:]  # optional client ids
    rc = 0
    for spec in load_specs():
        if only and spec["client_id"] not in only:
            continue
        print(f"[{spec['client_id']}]", file=sys.stderr)
        payload = run_client(spec)
        fname = f"{spec['client_id']}_role_feed_{dt.date.today().isoformat()}.json"
        if dry:
            os.makedirs("out", exist_ok=True)
            json.dump(payload, open(os.path.join("out", fname), "w"), indent=2)
            print(f"  wrote out/{fname}", file=sys.stderr)
        else:
            print(f"  emailed {fname} (Resend id {send_via_resend(spec, fname, payload)})", file=sys.stderr)
        if len(payload["model_errors"]) == len(MODELS):
            rc = 1  # every model failed: make the GitHub run go red
    return rc


if __name__ == "__main__":
    sys.exit(main())
