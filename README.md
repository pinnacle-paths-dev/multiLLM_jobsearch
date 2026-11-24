# Pinnacle Paths — Multi-LLM Role Feed

Runs Gemini and Grok with live web search against each client spec in `config/`,
merges and de-duplicates their roles, checks every link, and drops a dated JSON
feed into the client's Drive "Role Feeds" folder. The client's Claude scheduled
task (Mon/Thu 8am ET) reads that feed, runs the role-fit framework, verifies each
role on the company's own site, and sends the digest.

Schedule: Mon & Thu 11:15 UTC (6:15am ET winter / 7:15am ET summer).

## One-time setup (~15 minutes)

1. **Repo** — push this folder to a private GitHub repo.
2. **API keys** — repo Settings → Secrets and variables → Actions → New secret:
   - `GEMINI_API_KEY` — from https://aistudio.google.com/apikey
   - `XAI_API_KEY` — from https://console.x.ai
3. **Drive access (service account)**
   1. console.cloud.google.com → create/choose a project → enable **Google Drive API**.
   2. IAM & Admin → Service Accounts → Create → no roles needed → Keys → Add key → JSON.
   3. Paste the whole JSON file as secret `GDRIVE_SA_JSON`.
   4. In Drive, share each client's **Role Feeds** folder with the service account's
      email (…@…iam.gserviceaccount.com) as **Editor**.
4. **Client specs (secret)** — client data never goes in this repo. Add secret
   `CLIENT_SPECS_JSON` containing a JSON list of client specs (same shape as
   `config/example_client.json`). Nirbhay holds the current value.
5. **Optional model overrides** — Actions → Variables: `GEMINI_MODEL`, `GROK_MODEL`
   (defaults: gemini-3.5-flash, grok-4.7).
6. **Test** — Actions → "Multi-LLM role feed" → Run workflow → enter a client id.
   Check that client's Drive folder for `<client_id>_role_feed_<date>.json`.

## Adding a client

Add a new spec object (shape of `config/example_client.json`) to the
`CLIENT_SPECS_JSON` secret, create a "Role Feeds" folder in the client's Drive
folder, share it with the service account, put its ID in `drive_feed_folder_id`,
and add the feed-reading step to that client's Claude scheduled task.

## Local dry run

    pip install -r requirements.txt
    GEMINI_API_KEY=... XAI_API_KEY=... DRY_RUN=1 python feed.py <client_id>   # reads config/*.json locally, writes out/
    python tests/test_feed.py                                                  # offline test

## What the feed is (and isn't)

Discovery only. `link_status` is a basic HTTP check; JS-rendered ATS pages may
show "title not found in raw HTML". The Claude run is the hard gate: nothing reaches
an email unless Claude fetched the posting on the company's own site and saw it open.
