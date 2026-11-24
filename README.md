# Pinnacle Paths — Multi-LLM Role Feed

Runs Gemini and Grok with live web search, plus The Muse's jobs API, against each client spec, merges and
de-duplicates their roles, checks every link, and emails a dated JSON feed (as an
attachment, subject `[Role Feed] <client_id> <date>`) via Resend. The client's
Claude scheduled task (Mon/Thu 8am ET) finds that email in Gmail, runs the
role-fit framework, verifies each role on the company's own site, and sends the
digest.

Schedule: Mon & Thu 11:15 UTC (6:15am ET winter / 7:15am ET summer).

## One-time setup (~10 minutes)

1. **Secrets** — repo Settings → Secrets and variables → Actions → **Secrets** tab → New repository secret:
   - `GEMINI_API_KEY` — from https://aistudio.google.com/apikey
   - `XAI_API_KEY` — from https://console.x.ai
   - `MUSE_API_KEY` — from https://www.themuse.com/developers (optional: raises The Muse rate limit;
     The Muse runs only for clients whose spec has a `muse` block)
   - `RESEND_API_KEY` — from https://resend.com/api-keys (a "Sending access" key is enough)
   - `CLIENT_SPECS_JSON` — a JSON list of client specs (shape of
     `config/example_client.json`). Client data never goes in this repo.
2. **Variables** — same page → **Variables** tab → New repository variable:
   - `FEED_EMAIL_FROM` — a sender on your verified Resend domain, e.g. `Role Feed <jobs@yourdomain>`
   - `FEED_EMAIL_TO` — the inbox the Claude scheduled tasks read
   - optional `GEMINI_MODEL`, `GROK_MODEL` (defaults: gemini-3.5-flash, grok-4.7)
3. **Gmail filter (optional)** — filter `subject:"[Role Feed]"` → skip inbox, apply label `Role Feeds`.
4. **Test** — Actions → "Multi-LLM role feed" → Run workflow → enter a client id.
   A `[Role Feed] <client_id> <date>` email with a JSON attachment should arrive.

## Adding a client

Add a spec object to the `CLIENT_SPECS_JSON` secret and add the feed-reading step
(search Gmail for `[Role Feed] <client_id>`) to that client's Claude scheduled task.

## Local dry run

    pip install -r requirements.txt
    GEMINI_API_KEY=... XAI_API_KEY=... DRY_RUN=1 python feed.py <client_id>   # reads config/*.json locally, writes out/
    python tests/test_feed.py                                                  # offline test

## What the feed is (and isn't)

Discovery only. `link_status` is a basic HTTP check; JS-rendered ATS pages may
show "title not found in raw HTML". The Claude run is the hard gate: nothing reaches
a client unless Claude fetched the posting on the company's own site and saw it open.
