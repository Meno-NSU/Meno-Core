import json
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter
from typing import Literal, Dict, List
import datetime

arena_router = APIRouter(prefix="/v1/arena", tags=["arena"])

# Ensure resources dir exists
RESOURCES_DIR = Path("resources")
RESOURCES_DIR.mkdir(exist_ok=True)
VOTES_FILE = RESOURCES_DIR / "arena_votes.json"

class VoteRequest(BaseModel):
    model_a: str
    kb_a: str
    model_b: str
    kb_b: str
    winner: Literal["a", "b", "tie", "both_bad"]


def _load_votes() -> List[Dict]:
    if not VOTES_FILE.exists():
        return []
    try:
        content = VOTES_FILE.read_text(encoding="utf-8")
        if not content.strip():
            return []
        return json.loads(content)
    except Exception as e:
        print(f"Error loading votes: {e}")
        return []

def _save_votes(votes: List[Dict]):
    try:
        VOTES_FILE.write_text(json.dumps(votes, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Error saving votes: {e}")


def _calculate_elo(votes: List[Dict]) -> List[Dict]:
    # Base Elo
    INITIAL_ELO = 1200
    K_FACTOR = 32

    # Map of "model|kb" to stats
    stats = {}

    def get_stats(key: str):
        if key not in stats:
            stats[key] = {
                "elo": float(INITIAL_ELO),
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "matches": 0
            }
        return stats[key]

    for vote in votes:
        winner = vote.get("winner")
        # Ignore both_bad for Elo calculation, or treat as tie. Let's ignore it to not skew ratings.
        if winner == "both_bad":
            continue

        key_a = f"{vote['model_a']}|{vote['kb_a']}"
        key_b = f"{vote['model_b']}|{vote['kb_b']}"

        stat_a = get_stats(key_a)
        stat_b = get_stats(key_b)

        elo_a = stat_a["elo"]
        elo_b = stat_b["elo"]

        # Expected scores
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        # Actual scores
        if winner == "a":
            score_a, score_b = 1.0, 0.0
            stat_a["wins"] += 1
            stat_b["losses"] += 1
        elif winner == "b":
            score_a, score_b = 0.0, 1.0
            stat_a["losses"] += 1
            stat_b["wins"] += 1
        elif winner == "tie":
            score_a, score_b = 0.5, 0.5
            stat_a["ties"] += 1
            stat_b["ties"] += 1
        else:
            continue

        stat_a["matches"] += 1
        stat_b["matches"] += 1

        stat_a["elo"] = elo_a + K_FACTOR * (score_a - expected_a)
        stat_b["elo"] = elo_b + K_FACTOR * (score_b - expected_b)

    # Format result for leaderboard
    leaderboard = []
    for key, stat in stats.items():
        # Only include if they played at least one match
        model, kb = key.split("|", 1)
        win_rate = stat["wins"] / stat["matches"] if stat["matches"] > 0 else 0.0
        leaderboard.append({
            "model": model,
            "knowledge_base": kb,
            "elo": round(stat["elo"]),
            "matches": stat["matches"],
            "wins": stat["wins"],
            "losses": stat["losses"],
            "ties": stat["ties"],
            "win_rate": round(win_rate * 100, 1)
        })

    # Sort by Elo descending
    leaderboard.sort(key=lambda x: x["elo"], reverse=True)
    return leaderboard


@arena_router.post("/vote")
async def submit_vote(vote: VoteRequest):
    votes = _load_votes()
    vote_record = vote.model_dump()
    vote_record["timestamp"] = datetime.datetime.now().isoformat()
    votes.append(vote_record)
    _save_votes(votes)
    return {"status": "ok"}


@arena_router.get("/leaderboard")
async def get_leaderboard():
    votes = _load_votes()
    leaderboard = _calculate_elo(votes)
    return {"object": "list", "data": leaderboard}
