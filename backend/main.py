"""
ReclaimAI backend — FastAPI service exposing the trained recovery agent.

Endpoints:
  POST /decide           run the agent on a single (possibly in-flight) transaction, log to audit DB
  POST /decide-sequence  run the agent through a transaction's FULL multi-attempt recovery journey
  GET  /audit-log        recent live decisions from the SQLite audit trail
  GET  /metrics          precomputed agent-vs-baseline metrics (from batch simulation)
  GET  /simulation-log/{policy}   precomputed per-transaction audit trail for a policy
  GET  /health
"""
import json
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import environment as env
from agent import guardrails as gr
from agent.policy import QLearningPolicy
from agent.state import make_state
from backend import audit_db
from backend import llm

ROOT = Path(__file__).parent.parent

app = FastAPI(title="ReclaimAI — Agentic Revenue Recovery", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

policy = QLearningPolicy()
_q_path = ROOT / "agent" / "q_table.json"
if _q_path.exists():
    policy.load(str(_q_path))

audit_db.init_db()


class DecideRequest(BaseModel):
    transaction_id: str | None = None
    failure_reason: str
    amount_inr: float
    customer_risk: str = "low"
    attempt_number: int = 0
    hours_since_failure: float = 0.0
    hour_of_day: int = 12


def _amount_bucket(amount):
    if amount < 500:
        return "low"
    if amount <= 5000:
        return "mid"
    return "high"


@app.get("/health")
def health():
    return {"status": "ok", "q_table_loaded": _q_path.exists(), "states": len(policy.q)}


@app.post("/decide")
def decide(req: DecideRequest):
    if req.failure_reason not in gr.ACTIONS and req.failure_reason not in (
        "insufficient_funds", "bank_timeout", "risk_decline", "otp_failed", "card_expired", "cart_abandoned",
    ):
        raise HTTPException(400, f"unknown failure_reason: {req.failure_reason}")
    if req.customer_risk not in ("low", "medium", "high"):
        raise HTTPException(400, "customer_risk must be low|medium|high")

    amt_bucket = _amount_bucket(req.amount_inr)
    valid = gr.valid_actions(req.attempt_number, req.hours_since_failure, amt_bucket,
                              req.customer_risk, req.hour_of_day)
    state = make_state(req.failure_reason, amt_bucket, req.attempt_number, req.customer_risk,
                        req.hours_since_failure)
    action = policy.act_greedy(state, valid)
    qvals = policy.q_values(state, valid)
    state_seen = any(v is not None for v in qvals.values())

    decision_ctx = {
        "action": action, "failure_reason": req.failure_reason, "attempt_number": req.attempt_number,
        "customer_risk": req.customer_risk, "amount": req.amount_inr,
    }
    explanation = llm.explain_decision(decision_ctx)
    message = llm.customer_message(decision_ctx)

    record = {
        "transaction_id": req.transaction_id, "failure_reason": req.failure_reason,
        "amount_bucket": amt_bucket, "amount": req.amount_inr, "customer_risk": req.customer_risk,
        "attempt_number": req.attempt_number, "hours_since_failure": req.hours_since_failure,
        "action": action, "valid_actions": valid, "explanation": explanation, "customer_message": message,
    }
    audit_db.log_decision(record)

    considered = sorted(
        [{"action": a, "estimated_net_value_inr": (round(v, 2) if v is not None else None), "chosen": a == action}
         for a, v in qvals.items()],
        key=lambda r: (r["estimated_net_value_inr"] is None, -(r["estimated_net_value_inr"] or 0)),
    )

    return {
        "action": action,
        "valid_actions": valid,
        "explanation": explanation,
        "customer_message": message,
        "considered_actions": considered,
        "state_seen_in_training": state_seen,
        "guardrails_applied": {
            "max_attempts": gr.MAX_ATTEMPTS,
            "forced_stop_hours": gr.FORCED_STOP_HOURS,
            "quiet_hours": gr.is_quiet_hours(req.hour_of_day),
            "cooldown_hours_if_taken": gr.next_cooldown_hours(action),
        },
    }


class SequenceRequest(BaseModel):
    transaction_id: str | None = None
    failure_reason: str
    amount_inr: float
    customer_risk: str = "low"
    start_hour_of_day: int = 12
    seed: int | None = None


@app.post("/decide-sequence")
def decide_sequence(req: SequenceRequest):
    """Runs a transaction through the FULL recovery journey — every attempt,
    cooldown, and either an eventual success or a forced stop — instead of
    just one isolated decision. This is what actually exercises the
    guardrails (cooldowns, quiet hours, max attempts, forced write-off)."""
    if req.customer_risk not in ("low", "medium", "high"):
        raise HTTPException(400, "customer_risk must be low|medium|high")

    amt_bucket = _amount_bucket(req.amount_inr)
    rng = random.Random(req.seed if req.seed is not None else random.randint(0, 2**31))

    attempt_number = 0
    hours_since_failure = 0.0
    hour_of_day = req.start_hour_of_day
    steps = []
    recovered = False

    while True:
        valid = gr.valid_actions(attempt_number, hours_since_failure, amt_bucket, req.customer_risk, hour_of_day)
        state = make_state(req.failure_reason, amt_bucket, attempt_number, req.customer_risk, hours_since_failure)
        action = policy.act_greedy(state, valid)

        decision_ctx = {
            "action": action, "failure_reason": req.failure_reason, "attempt_number": attempt_number,
            "customer_risk": req.customer_risk, "amount": req.amount_inr,
        }
        explanation = llm.explain_decision(decision_ctx)

        if action == "stop":
            steps.append({
                "attempt": attempt_number, "hour_of_day": hour_of_day, "action": action,
                "outcome": "written_off", "explanation": explanation,
            })
            break

        success, reward = env.step(req.failure_reason, action, attempt_number, req.customer_risk, req.amount_inr, rng)
        steps.append({
            "attempt": attempt_number, "hour_of_day": hour_of_day, "action": action,
            "outcome": "recovered" if success else "no_response",
            "customer_message": llm.customer_message(decision_ctx),
            "explanation": explanation,
        })

        if success:
            recovered = True
            break

        attempt_number += 1
        cooldown = gr.next_cooldown_hours(action)
        hours_since_failure += cooldown
        hour_of_day = (hour_of_day + cooldown) % 24

        if attempt_number >= gr.MAX_ATTEMPTS or hours_since_failure >= gr.FORCED_STOP_HOURS:
            steps.append({
                "attempt": attempt_number, "hour_of_day": hour_of_day, "action": "stop",
                "outcome": "written_off", "explanation": "Guardrail forced stop: max attempts or 72h limit reached.",
            })
            break

    return {
        "recovered": recovered,
        "total_attempts": sum(1 for s in steps if s["action"] != "stop"),
        "steps": steps,
    }


@app.get("/audit-log")
def audit_log(limit: int = 100):
    return audit_db.recent_decisions(limit)


@app.get("/metrics")
def metrics():
    path = ROOT / "agent" / "results" / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "run `python -m agent.simulate` first")
    return json.loads(path.read_text())


@app.get("/simulation-log/{policy_name}")
def simulation_log(policy_name: str, limit: int = 200):
    path = ROOT / "agent" / "results" / "audit_log.json"
    if not path.exists():
        raise HTTPException(404, "run `python -m agent.simulate` first")
    data = json.loads(path.read_text())
    if policy_name not in data:
        raise HTTPException(404, f"unknown policy: {policy_name}. known: {list(data.keys())}")
    return data[policy_name][:limit]


_frontend_dir = ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
