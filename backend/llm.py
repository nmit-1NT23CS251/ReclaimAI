"""
Pluggable explanation layer. Every agent decision gets a human-readable
rationale and, for contact actions, a draft customer message. If a
GROQ_API_KEY is set, we call Groq's Llama 3.3 70B (same provider used by
SafarAI's Destination Guide feature) for fluent text. If not, a templated
fallback produces an equivalent (less fluent, but fully functional and
still audit-worthy) explanation — the system never depends on a live LLM
call to run or to be demoable.
"""
import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

ACTION_BLURB = {
    "retry_now": "silently retried the payment through the gateway with no customer contact",
    "retry_with_reminder": "sent the customer a reminder with a retry link",
    "offer_discount": "offered the customer a small incentive alongside a retry link",
    "escalate_human": "handed this off to a human support agent",
    "stop": "stopped pursuing recovery and wrote the transaction off",
}


def _groq_available():
    return bool(os.environ.get("GROQ_API_KEY"))


def _call_groq(system_prompt, user_prompt, max_tokens=120):
    api_key = os.environ.get("GROQ_API_KEY")
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        },
        timeout=8,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _template_explanation(decision):
    action = decision["action"]
    reason = decision["failure_reason"]
    attempt = decision["attempt_number"]
    blurb = ACTION_BLURB.get(action, action)
    return (
        f"Attempt {attempt + 1}: for a {reason.replace('_', ' ')} failure "
        f"({decision['customer_risk']} risk customer, ₹{decision['amount']:.0f}), "
        f"the agent {blurb}. Guardrails allowed this action at this stage; "
        f"stricter/costlier actions were withheld until cheaper ones were tried."
    )


def _template_customer_message(decision):
    action = decision["action"]
    if action == "retry_with_reminder":
        return ("Hi! We noticed your recent payment didn't go through. "
                "No worries — you can complete it anytime using the link below.")
    if action == "offer_discount":
        return ("Hi! Your payment didn't go through, but we'd still love to have you. "
                "Here's a small discount if you complete it in the next 24 hours.")
    if action == "escalate_human":
        return ("Hi! We're having trouble processing your payment — one of our support "
                "specialists will reach out shortly to help sort it out.")
    return ""


def explain_decision(decision: dict) -> str:
    """decision: {action, failure_reason, attempt_number, customer_risk, amount}"""
    if not _groq_available():
        return _template_explanation(decision)
    try:
        system = ("You write one-sentence, plain-English audit-log explanations for an "
                   "automated payment-recovery agent's decisions. Be concise and factual.")
        user = (f"Decision: {decision['action']} | failure reason: {decision['failure_reason']} | "
                f"attempt number: {decision['attempt_number'] + 1} | customer risk: {decision['customer_risk']} | "
                f"amount: INR {decision['amount']:.0f}. Explain briefly why this action makes sense here.")
        return _call_groq(system, user)
    except Exception:
        return _template_explanation(decision)


def customer_message(decision: dict) -> str:
    if decision["action"] not in ("retry_with_reminder", "offer_discount", "escalate_human"):
        return ""
    if not _groq_available():
        return _template_customer_message(decision)
    try:
        system = ("You write short, warm, non-pushy customer messages (2-3 sentences) for a "
                   "payment-recovery flow. Never use guilt or urgency manipulation.")
        user = (f"Write a customer-facing message for action '{decision['action']}' after a "
                f"{decision['failure_reason'].replace('_', ' ')} payment failure of about "
                f"INR {decision['amount']:.0f}.")
        return _call_groq(system, user, max_tokens=100)
    except Exception:
        return _template_customer_message(decision)
