# 99plus — Match Intelligence Platform

## What's in this package
- `index.html` — the full frontend: fullscreen YouTube ambient background, low-friction
  nickname-only signup (LocalStorage session), the live prediction board (client-side
  Poisson odds engine), and an in-page Admin control room (nav → "admin", demo passcode `9999`).
- `odds_engine.py` — the server-side prediction/odds engine meant to run on a schedule
  (cron / APScheduler / serverless cron). Fetches fixtures + form + H2H from a provider
  like API-Football, runs the same Poisson expected-goals model, and publishes odds —
  skipping any fixture an admin has manually overridden.

## Running the frontend locally
No build step. Just open `index.html` in a browser, or serve it:
    python3 -m http.server 8000
then visit http://localhost:8000

## Running the backend
    pip install requests
    export API_FOOTBALL_KEY="your_key_here"
    python3 odds_engine.py

Note: `db_get_override` / `db_upsert_prediction` are stubs — wire them to your real
database before running this against production data.

## Known limitation (by design, for this demo)
`index.html` uses browser LocalStorage for accounts/overrides, so the admin panel only
sees users signed up in that same browser. A real multi-user launch needs a shared
backend/database — see the "Scheduling this in production" notes at the bottom of
odds_engine.py, and swap the stub DB functions for real ones.
