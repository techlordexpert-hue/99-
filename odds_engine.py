"""
99plus — Automated Prediction & Odds Engine
=============================================

This module is the server-side counterpart to the JS engine embedded in the
frontend (99plus.html). It's meant to run on a schedule (cron / APScheduler /
a serverless scheduled function) and do three jobs:

  1. fetch_fixtures()   -> pull next fixtures + rolling form + H2H from a
                            live provider (API-Football is used as the example).
  2. calculate_prediction() -> turn form metrics into Home/Draw/Away
                            probabilities via a Poisson expected-goals model,
                            then into decimal odds with a bookmaker margin.
  3. publish()          -> write predictions to the database, skipping any
                            fixture an admin has manually overridden.

Nothing here needs a GPU or exotic infra — it's a scheduled Python job plus a
Postgres table. Swap the `requests` calls for your real provider credentials
and the `db_*` stubs for your actual database layer (Postgres/MongoDB/etc).
"""

from __future__ import annotations
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests  # pip install requests


# ---------------------------------------------------------------------------
# 1. CONFIG — set these via environment variables, never hardcode keys
# ---------------------------------------------------------------------------
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_FOOTBALL_HOST = "v3.football.api-sports.io"
BOOKMAKER_MARGIN = 0.06        # 6% overround baked into published odds
HOME_ADVANTAGE = 1.12          # multiplier applied to home team's expected goals
MAX_GOALS = 8                  # truncation point for the Poisson score grid
VALUE_BET_EDGE_THRESHOLD = 0.05  # min. edge over market to flag "value bet"


# ---------------------------------------------------------------------------
# 2. DATA SHAPES
# ---------------------------------------------------------------------------
@dataclass
class TeamForm:
    """Rolling-form inputs, typically computed from a team's last 5-10 matches."""
    attack_strength: float   # goals scored / league-average goals scored
    defense_strength: float  # goals conceded / league-average goals conceded


@dataclass
class Fixture:
    fixture_id: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    home_form: TeamForm
    away_form: TeamForm
    h2h_home_win_rate: Optional[float] = None  # optional extra signal, 0-1
    market_odds: dict = field(default_factory=dict)  # {"home":.., "draw":.., "away":..} for value-bet comparison


@dataclass
class Prediction:
    fixture_id: str
    probs: dict          # {"home":0.51, "draw":0.24, "away":0.25}
    odds: dict            # {"home":1.75, "draw":3.80, "away":3.60}
    lambda_home: float
    lambda_away: float
    value_bet: Optional[str] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 3. FETCH — pull fixtures, form, and head-to-head from the live provider
# ---------------------------------------------------------------------------
def _api_get(path: str, params: dict) -> dict:
    resp = requests.get(
        f"https://{API_FOOTBALL_HOST}/{path}",
        headers={"x-apisports-key": API_FOOTBALL_KEY},
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_next_fixtures(league_id: int, season: int, next_n: int = 15) -> list[dict]:
    """GET /fixtures?league=..&season=..&next=.. -> upcoming fixtures for a league."""
    data = _api_get("fixtures", {"league": league_id, "season": season, "next": next_n})
    return data.get("response", [])


def fetch_team_form(team_id: int, last_n: int = 8) -> TeamForm:
    """
    GET /teams/statistics (or /fixtures?team=..&last=..) and derive rolling
    attack/defense strength relative to league average. Simplified here —
    in production, compute goals-for/against per game vs league averages.
    """
    data = _api_get("teams/statistics", {"team": team_id, "last": last_n})
    stats = data.get("response", {})
    goals_for_avg = float(stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.2) or 1.2)
    goals_against_avg = float(stats.get("goals", {}).get("against", {}).get("average", {}).get("total", 1.2) or 1.2)
    league_avg_goals = 1.35  # pull this per-league from provider or maintain a lookup table
    return TeamForm(
        attack_strength=goals_for_avg / league_avg_goals,
        defense_strength=goals_against_avg / league_avg_goals,
    )


def fetch_h2h_home_win_rate(team_a_id: int, team_b_id: int, last_n: int = 10) -> Optional[float]:
    """GET /fixtures/headtohead -> fraction of recent meetings the home side won."""
    data = _api_get("fixtures/headtohead", {"h2h": f"{team_a_id}-{team_b_id}", "last": last_n})
    matches = data.get("response", [])
    if not matches:
        return None
    home_wins = sum(1 for m in matches if m["teams"]["home"]["winner"] is True)
    return home_wins / len(matches)


def build_fixture_batch(league_id: int, season: int) -> list[Fixture]:
    """Orchestrates the three calls above into ready-to-score Fixture objects."""
    raw_fixtures = fetch_next_fixtures(league_id, season)
    fixtures: list[Fixture] = []
    for f in raw_fixtures:
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]
        fixtures.append(Fixture(
            fixture_id=str(f["fixture"]["id"]),
            home_team=f["teams"]["home"]["name"],
            away_team=f["teams"]["away"]["name"],
            kickoff_utc=datetime.fromisoformat(f["fixture"]["date"]),
            home_form=fetch_team_form(home_id),
            away_form=fetch_team_form(away_id),
            h2h_home_win_rate=fetch_h2h_home_win_rate(home_id, away_id),
        ))
        time.sleep(0.3)  # respect provider rate limits
    return fixtures


# ---------------------------------------------------------------------------
# 4. MODEL — Poisson expected-goals -> probabilities -> decimal odds
# ---------------------------------------------------------------------------
def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def calculate_prediction(fixture: Fixture) -> Prediction:
    """
    Core statistical function.

    Expected goals (lambda) for each side = attack_strength(team) *
    defense_strength(opponent) * home_advantage (home side only). H2H is
    folded in as a small nudge on the home lambda so it influences the
    result without dominating the form-based signal.

    We then build the full scoreline probability grid (0..MAX_GOALS goals
    each side) via the Poisson PMF, and sum cells into Home/Draw/Away.
    """
    lambda_home = fixture.home_form.attack_strength * fixture.away_form.defense_strength * HOME_ADVANTAGE
    lambda_away = fixture.away_form.attack_strength * fixture.home_form.defense_strength

    if fixture.h2h_home_win_rate is not None:
        # Nudge home lambda by up to +/-8% based on historical H2H home win rate vs. a neutral 45%.
        nudge = (fixture.h2h_home_win_rate - 0.45) * 0.18
        lambda_home *= (1 + nudge)

    p_home = p_draw = p_away = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = _poisson_pmf(lambda_home, i) * _poisson_pmf(lambda_away, j)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away  # normalize the truncated grid back to 1.0
    probs = {"home": p_home / total, "draw": p_draw / total, "away": p_away / total}
    odds = _probs_to_decimal_odds(probs, BOOKMAKER_MARGIN)
    value_bet = _find_value_bet(probs, fixture.market_odds)

    return Prediction(
        fixture_id=fixture.fixture_id,
        probs=probs,
        odds=odds,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        value_bet=value_bet,
    )


def _probs_to_decimal_odds(probs: dict, margin: float) -> dict:
    """
    Fair decimal odds = 1 / true_probability. A bookmaker never posts fair
    odds — it applies an overround (margin) so the implied probabilities sum
    to > 100%. We scale each probability up by (1 + margin) before inverting.
    """
    scale = 1 + margin
    return {k: round(1 / (v * scale), 2) for k, v in probs.items()}


def _find_value_bet(probs: dict, market_odds: dict) -> Optional[str]:
    """
    A "value bet" is an outcome where our model's fair probability exceeds
    the market's implied probability by more than VALUE_BET_EDGE_THRESHOLD —
    i.e. the market is pricing that outcome too generously relative to our model.
    """
    best_edge = 0.0
    best_outcome = None
    for outcome, odds in market_odds.items():
        if not odds or odds <= 1:
            continue
        market_implied = 1 / odds
        edge = probs.get(outcome, 0) - market_implied
        if edge > VALUE_BET_EDGE_THRESHOLD and edge > best_edge:
            best_edge, best_outcome = edge, outcome
    return best_outcome


# ---------------------------------------------------------------------------
# 5. PUBLISH — write to DB, respecting admin overrides
# ---------------------------------------------------------------------------
def db_get_override(fixture_id: str) -> Optional[dict]:
    """Stub: SELECT * FROM prediction_overrides WHERE fixture_id = %s"""
    raise NotImplementedError("Wire this up to your database layer")


def db_upsert_prediction(fixture: Fixture, prediction: Prediction) -> None:
    """Stub: INSERT ... ON CONFLICT (fixture_id) DO UPDATE ..."""
    raise NotImplementedError("Wire this up to your database layer")


def publish_predictions(fixtures: list[Fixture]) -> list[Prediction]:
    published = []
    for fixture in fixtures:
        override = db_get_override(fixture.fixture_id)
        if override:
            # Admin's manual numbers win outright — the model never overwrites them.
            continue
        prediction = calculate_prediction(fixture)
        db_upsert_prediction(fixture, prediction)
        published.append(prediction)
    return published


# ---------------------------------------------------------------------------
# 6. ENTRY POINT — this is what the cron job / scheduled function calls
# ---------------------------------------------------------------------------
def run_prediction_cycle(league_id: int, season: int) -> None:
    fixtures = build_fixture_batch(league_id, season)
    results = publish_predictions(fixtures)
    print(f"[{datetime.now(timezone.utc).isoformat()}] published {len(results)} predictions "
          f"for league {league_id}")


if __name__ == "__main__":
    # Example: Premier League (API-Football league_id 39), season 2026
    run_prediction_cycle(league_id=39, season=2026)


"""
Scheduling this in production
------------------------------
Option A — plain cron (simplest, one server):
    */30 * * * *  cd /srv/99plus && /usr/bin/python3 odds_engine.py >> logs/odds.log 2>&1
    Runs every 30 min; fine since fixtures/form don't change minute-to-minute.

Option B — APScheduler (if the engine lives inside a long-running app process):
    from apscheduler.schedulers.background import BackgroundScheduler
    sched = BackgroundScheduler()
    sched.add_job(run_prediction_cycle, "interval", minutes=30, args=[39, 2026])
    sched.start()

Option C — serverless scheduled function (AWS EventBridge -> Lambda, or
    Vercel/Cloudflare Cron Triggers) if you don't want to manage a server at all.

Whichever you choose, keep fetch_* calls idempotent and rate-limit-aware
(API-Football's free tier is 100 req/day), and always let db_get_override()
short-circuit publish_predictions() so admin edits are never clobbered by
the next scheduled run.
"""
