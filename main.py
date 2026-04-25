"""
ShieldPay - FastAPI Backend (agentic version)
Run from project root: uvicorn main:app --reload --port 8000

Endpoints:
  POST /analyze              - Claude analyses a text checkout, streaming (legacy)
  POST /analyze-image-upload - Agentic image analysis with tool use (NEW)
  POST /create-card          - Direct bunq card creation (still available)
  POST /cancel-card          - Cancels a bunq card (called by the pay button)
  GET  /get-card/{card_id}   - Fetch real card details from bunq
  GET  /default-card         - Get-or-create the user's default virtual card
  GET  /allowed-card-names   - bunq permitted card names
  GET  /health               - Sanity check
"""

import os
import json
import base64
import requests
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import anthropic
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

# Config
BUNQ_API_URL         = os.getenv("BUNQ_API_URL", "https://public-api.sandbox.bunq.com/v1")
SESSION_TOKEN        = os.getenv("SESSION_TOKEN")
USER_ID              = os.getenv("USER_ID")
MONETARY_ACCOUNT_ID  = os.getenv("MONETARY_ACCOUNT_ID")
KEY_PATH             = os.getenv("BUNQ_INSTALLATION_KEY_PATH", "./bunq-user-1/installation.key")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")
CARD_NAME            = os.getenv("CARD_NAME", "Card Holder")

# Startup validation
missing = [k for k, v in {
    "SESSION_TOKEN": SESSION_TOKEN,
    "USER_ID": USER_ID,
    "MONETARY_ACCOUNT_ID": MONETARY_ACCOUNT_ID,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
}.items() if not v]
if missing:
    raise RuntimeError(f"Missing .env variables: {', '.join(missing)}")

# RSA signing
with open(KEY_PATH, "rb") as f:
    _private_key = serialization.load_pem_private_key(f.read(), password=None)

def _sign(body_str: str) -> str:
    sig = _private_key.sign(body_str.encode(), padding.PKCS1v15(), hashes.SHA256())
    return base64.b64encode(sig).decode()

def bunq_headers(body_str: str = "") -> dict:
    base = {
        "Cache-Control": "no-cache",
        "User-Agent": "shieldpay-hackathon",
        "X-Bunq-Client-Authentication": SESSION_TOKEN,
        "X-Bunq-Language": "en_US",
        "X-Bunq-Region": "nl_NL",
        "X-Bunq-Geolocation": "0 0 0 0 NL",
        "Content-Type": "application/json",
    }
    if body_str:
        base["X-Bunq-Client-Signature"] = _sign(body_str)
    return base

# Anthropic client
ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-6"

# FastAPI app
app = FastAPI(title="ShieldPay API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class CheckoutContext(BaseModel):
    merchant_name: str
    scenario: str
    amount: float
    description: str

class CreateCardRequest(BaseModel):
    scenario: str
    amount: float
    limit: float
    expiry_days: int | None = None

class CancelCardRequest(BaseModel):
    card_id: int

# ---------- Tool implementations ----------

EXTRACT_SYSTEM = """You extract checkout details from a screenshot.
Return ONLY a JSON object, no prose, no markdown:
{
  "merchant": "<name>",
  "amount": <number>,
  "currency": "<3-letter code>",
  "summary": "<one short sentence describing what is being purchased>",
  "is_subscription_signup": <true|false>
}
If the image is unreadable, set merchant to "unknown" and amount to 0."""


def tool_extract_checkout_details(image_b64: str, media_type: str) -> dict:
    """Runs a dedicated vision call to pull structured checkout data."""
    resp = ai.messages.create(
        model=MODEL,
        max_tokens=400,
        system=EXTRACT_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": image_b64
                }},
                {"type": "text", "text": "Extract the checkout details."}
            ]
        }]
    )
    text = resp.content[0].text.strip()
    return _safe_json_parse(text, fallback={
        "merchant": "unknown", "amount": 0, "currency": "EUR",
        "summary": "Could not read image", "is_subscription_signup": False
    })


def _safe_json_parse(text: str, fallback):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return fallback
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return fallback


TRUSTED_MERCHANTS = {
    "netflix", "spotify", "amazon", "apple", "google", "microsoft",
    "uber", "bol.com", "albert heijn", "hema", "ikea", "zalando", "asos"
}
SUSPICIOUS_KEYWORDS = {
    "gadgets4u", "deals4less", "freeshippingnow", "buynowcheap"
}


def tool_lookup_merchant_reputation(merchant_name: str) -> dict:
    n = (merchant_name or "").lower()
    for t in TRUSTED_MERCHANTS:
        if t in n:
            return {"merchant": merchant_name, "reputation": "trusted",
                    "note": "Well known reputable merchant."}
    for s in SUSPICIOUS_KEYWORDS:
        if s in n:
            return {"merchant": merchant_name, "reputation": "suspicious",
                    "note": "Matches patterns associated with risky merchants."}
    return {"merchant": merchant_name, "reputation": "unknown",
            "note": "Not in our trusted or suspicious list. Treat with caution."}


# Mock transaction history. In production this would hit bunq's payment API.
MOCK_TRANSACTIONS = [
    {"merchant": "Netflix",      "amount": 15.99, "date": "2025-09-15"},
    {"merchant": "Albert Heijn", "amount": 42.30, "date": "2025-09-10"},
    {"merchant": "Spotify",      "amount": 10.99, "date": "2025-08-28"},
    {"merchant": "Bol.com",      "amount": 67.50, "date": "2025-08-22"},
]


def tool_get_user_recent_transactions(merchant_name: str = "") -> dict:
    if not merchant_name:
        return {"recent_transactions": MOCK_TRANSACTIONS}
    n = merchant_name.lower()
    matches = [t for t in MOCK_TRANSACTIONS
               if n in t["merchant"].lower() or t["merchant"].lower() in n]
    return {
        "merchant_searched": merchant_name,
        "paid_before": len(matches) > 0,
        "matches": matches,
    }


def tool_create_shield_card(scenario: str, limit: float,
                            expiry_days: int | None = None) -> dict:
    if scenario == "trusted_merchant":
        return {"created": False, "reason": "Trusted merchant. No card needed."}
    return _create_bunq_card(scenario, limit, expiry_days)


def tool_notify_user(message: str, risk: str) -> dict:
    return {"message": message, "risk": risk}


DEFAULT_CARD_TAG = "MAIN CARD"
DEFAULT_CARD_LIMIT = 5000.0
_default_card_cache: dict | None = None


def _create_bunq_card(scenario: str, limit: float,
                      expiry_days: int | None = None) -> dict:
    """Creates a bunq virtual shield card and applies a spending limit."""
    labels = {
        "free_trial":       "TRIAL SHIELD",
        "unknown_merchant": "ONE-TIME CARD",
        "trusted_merchant": "SHIELD CARD",
    }
    second_line = labels.get(scenario, "SHIELD CARD")

    card_id, card_data, error = _post_card_to_bunq(second_line, limit)
    if error:
        return {"created": False, "error": error}

    if expiry_days:
        display_expiry = (datetime.utcnow() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
    else:
        display_expiry = card_data.get("expiry_date", "2030-05-31")

    return {
        "created": True,
        "card_id": card_id,
        "second_line": second_line,
        "name_on_card": CARD_NAME,
        "limit": limit,
        "currency": "EUR",
        "scenario": scenario,
        "expiry_date": display_expiry,
        "status": "ACTIVE",
        "masked_number": f"**** **** **** {str(card_id)[-4:].zfill(4)}",
    }


def _post_card_to_bunq(second_line: str, limit: float):
    """POSTs a card and PUTs its limit. Returns (card_id, card_data, error)."""
    card_body = {
        "second_line": second_line,
        "name_on_card": CARD_NAME,
        "type": "MASTERCARD",
        "product_type": "MASTERCARD_DEBIT",
        "pin_code_assignment": [{
            "type": "PRIMARY",
            "pin_code": os.getenv("CARD_PIN", "1234"),
            "monetary_account_id": int(MONETARY_ACCOUNT_ID),
        }],
    }
    body_str = json.dumps(card_body)
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card-debit"
    resp = requests.post(url, headers=bunq_headers(body_str), data=body_str)
    if resp.status_code not in (200, 201):
        return None, {}, f"bunq error: {resp.text}"

    card_id, card_data = _parse_card_id(resp.json())
    if not card_id:
        return None, {}, "Could not parse card ID from bunq"

    limit_body = json.dumps({
        "card_limit": {"value": f"{limit:.2f}", "currency": "EUR"},
        "status": "ACTIVE",
    })
    put_url = f"{BUNQ_API_URL}/user/{USER_ID}/card/{card_id}"
    limit_resp = requests.put(put_url, headers=bunq_headers(limit_body), data=limit_body)
    if limit_resp.status_code != 200:
        return None, {}, f"bunq limit error: {limit_resp.text}"

    return card_id, card_data, None


def _parse_card_id(bunq_response: dict):
    response_list = bunq_response.get("Response", [])
    for item in response_list:
        if "CardDebit" in item:
            data = item["CardDebit"]
            return data.get("id"), data
        if "Id" in item:
            return item["Id"]["id"], {}
    return None, {}


# ---------- Default card (get-or-create) ----------

def _list_user_cards() -> list[dict]:
    """Lists all cards on the bunq user account."""
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card?count=200"
    resp = requests.get(url, headers=bunq_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"bunq list error: {resp.text}")
    cards = []
    for item in resp.json().get("Response", []):
        for key in ("CardDebit", "Card", "CardCredit"):
            if key in item:
                cards.append(item[key])
                break
    return cards


def _to_default_card_record(bunq_card: dict) -> dict:
    card_id = bunq_card.get("id")
    return {
        "card_id": card_id,
        "second_line": bunq_card.get("second_line") or DEFAULT_CARD_TAG,
        "name_on_card": bunq_card.get("name_on_card") or CARD_NAME,
        "masked_number": f"**** **** **** {str(card_id)[-4:].zfill(4)}",
        "expiry_date": bunq_card.get("expiry_date", "2030-05-31"),
        "status": bunq_card.get("status", "ACTIVE"),
    }


def _get_or_create_default_card() -> dict:
    """Returns the default virtual card. Creates one if missing.

    Sets created_now=True only on the request that actually performed the
    creation. Subsequent requests in the same process hit a cache and report
    created_now=False.
    """
    global _default_card_cache
    if _default_card_cache:
        return {**_default_card_cache, "created_now": False}

    for card in _list_user_cards():
        if (card.get("second_line") == DEFAULT_CARD_TAG
                and card.get("status") == "ACTIVE"):
            record = _to_default_card_record(card)
            _default_card_cache = record
            return {**record, "created_now": False}

    card_id, card_data, error = _post_card_to_bunq(DEFAULT_CARD_TAG, DEFAULT_CARD_LIMIT)
    if error:
        raise HTTPException(status_code=502, detail=error)
    record = {
        "card_id": card_id,
        "second_line": DEFAULT_CARD_TAG,
        "name_on_card": CARD_NAME,
        "masked_number": f"**** **** **** {str(card_id)[-4:].zfill(4)}",
        "expiry_date": card_data.get("expiry_date", "2030-05-31"),
        "status": "ACTIVE",
    }
    _default_card_cache = record
    return {**record, "created_now": True}


# ---------- Agent loop ----------

AGENT_TOOLS = [
    {
        "name": "extract_checkout_details",
        "description": (
            "Read the checkout screenshot the user just uploaded. "
            "Returns merchant, amount, currency, summary, is_subscription_signup. "
            "Always call this first."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "lookup_merchant_reputation",
        "description": "Look up whether a merchant is trusted, suspicious, or unknown.",
        "input_schema": {
            "type": "object",
            "properties": {"merchant_name": {"type": "string"}},
            "required": ["merchant_name"],
        },
    },
    {
        "name": "get_user_recent_transactions",
        "description": (
            "Check whether the user has paid this merchant before. "
            "Use this when reputation is unknown to gain extra signal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"merchant_name": {"type": "string"}},
            "required": ["merchant_name"],
        },
    },
    {
        "name": "create_shield_card",
        "description": (
            "Create a virtual bunq card. "
            "Use scenario='free_trial' with limit=0.01 and expiry_days=29 for trials that auto-charge later. "
            "Use scenario='unknown_merchant' with limit set to the exact purchase amount for unknown but identifiable merchants. "
            "Do NOT call this for trusted merchants; skip straight to notify_user. "
            "Do NOT call this when the checkout looks fraudulent or has no identifiable merchant or amount; "
            "in that case skip straight to notify_user with risk='high'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {"type": "string",
                             "enum": ["free_trial", "unknown_merchant"]},
                "limit": {"type": "number"},
                "expiry_days": {"type": ["integer", "null"]},
            },
            "required": ["scenario", "limit"],
        },
    },
    {
        "name": "notify_user",
        "description": (
            "Send the final 1-2 sentence explanation to the user. "
            "Always call this LAST. risk must be high, medium, or low."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "risk": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["message", "risk"],
        },
    },
]

AGENT_SYSTEM = """You are ShieldPay, an AI payment protection agent.

A user just uploaded a checkout screenshot. Decide the safest way to pay using your tools.

Workflow:
1. Call extract_checkout_details first.
2. Call lookup_merchant_reputation with the merchant name.
3. If reputation is "unknown", call get_user_recent_transactions for extra signal.
4. Decide a strategy. Pick exactly one:
   A) TRUSTED merchant with a reasonable amount: skip card creation. Call notify_user with risk="low".
   B) FREE TRIAL that will auto-charge later: create_shield_card with scenario="free_trial", limit=0.01, expiry_days=29. Then notify_user with risk="medium".
   C) UNKNOWN but identifiable merchant (real name and amount visible): create_shield_card with scenario="unknown_merchant", limit equal to the exact purchase amount. Then notify_user with risk="medium".
   D) FRAUDULENT or UNIDENTIFIABLE checkout (no merchant name visible, no amount visible, or merchant flagged as suspicious): DO NOT create a card. The user should not pay at all. Call notify_user with risk="high" and tell them not to proceed.

A 1-cent shield card is not a substitute for refusing to pay. If the page itself looks fake, choose strategy D.

Be decisive. Do not narrate or explain between tool calls. The UI shows tool progress directly. Your only user-facing text is the message you pass to notify_user."""


def _serialize_assistant(content) -> list:
    """Convert SDK content blocks back into dict form for the next API call."""
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id,
                        "name": b.name, "input": b.input})
    return out


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _run_tool(name: str, inp: dict, image_b64: str, media_type: str) -> dict:
    if name == "extract_checkout_details":
        return tool_extract_checkout_details(image_b64, media_type)
    if name == "lookup_merchant_reputation":
        return tool_lookup_merchant_reputation(inp["merchant_name"])
    if name == "get_user_recent_transactions":
        return tool_get_user_recent_transactions(inp.get("merchant_name", ""))
    if name == "create_shield_card":
        return tool_create_shield_card(
            inp["scenario"], inp["limit"], inp.get("expiry_days")
        )
    if name == "notify_user":
        return tool_notify_user(inp["message"], inp["risk"])
    return {"error": f"Unknown tool: {name}"}


def run_shieldpay_agent(image_b64: str, media_type: str, hint):
    """Runs the agentic loop and yields SSE events for the frontend."""
    user_msg = (
        "The user has uploaded a checkout screenshot. "
        f"User hint: {hint or 'none'}. "
        "Use your tools to analyze and protect this payment."
    )
    messages = [{"role": "user", "content": user_msg}]

    max_turns = 8
    for _ in range(max_turns):
        try:
            with ai.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=AGENT_SYSTEM,
                tools=AGENT_TOOLS,
                messages=messages,
            ) as stream:
                for event in stream:
                    et = getattr(event, "type", None)
                    if et == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            yield _sse({"type": "tool_start",
                                        "id": block.id, "name": block.name})
                    elif et == "content_block_delta":
                        delta = event.delta
                        if getattr(delta, "type", None) == "text_delta":
                            yield _sse({"type": "text", "content": delta.text})

                final = stream.get_final_message()
        except Exception as e:
            yield _sse({"type": "error", "content": str(e)})
            return

        if final.stop_reason != "tool_use":
            yield _sse({"type": "done"})
            return

        tool_results = []
        for block in final.content:
            if block.type != "tool_use":
                continue
            try:
                result = _run_tool(block.name, block.input, image_b64, media_type)
            except Exception as e:
                result = {"error": str(e)}

            yield _sse({
                "type": "tool_result",
                "id": block.id,
                "name": block.name,
                "input": block.input,
                "result": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "assistant",
                         "content": _serialize_assistant(final.content)})
        messages.append({"role": "user", "content": tool_results})

    yield _sse({"type": "error", "content": "Agent loop exceeded max turns"})


# ---------- Endpoints ----------

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/")


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# Legacy text-only analyze (unchanged contract, kept for backwards compat)
SYSTEM_PROMPT = """You are ShieldPay, an AI payment protection agent.
Your job is to analyse online checkout situations and decide the safest way to pay.

You will receive a checkout context and must respond with:
1. A brief real-time analysis (think out loud, 2-4 sentences, conversational)
2. Then a JSON block with your decision

Scenarios you handle:
- free_trial: Merchant is offering a free trial that auto-charges later
- unknown_merchant: Merchant is unfamiliar or untrusted
- trusted_merchant: Well-known reputable merchant with small purchase

Your decisions:
- free_trial -> create a virtual card with EUR 0.01 limit
- unknown_merchant -> create a one-time virtual card capped at exact purchase amount
- trusted_merchant -> no card needed

After your analysis, output EXACTLY this JSON block (no markdown fences):
DECISION_JSON:
{
  "risk": "high" | "medium" | "low",
  "action": "create_card" | "no_action",
  "card_type": "trial_shield" | "one_time" | null,
  "limit": <number in EUR or null>,
  "expiry_days": <number or null>,
  "reason": "<one sentence summary>"
}

Be concise and confident."""


def _fallback_decision(scenario: str, amount: float) -> dict:
    if scenario == "free_trial":
        return {"risk": "high", "action": "create_card", "card_type": "trial_shield",
                "limit": 0.01, "expiry_days": 29,
                "reason": "Free trial detected, shield card created."}
    if scenario == "unknown_merchant":
        return {"risk": "medium", "action": "create_card", "card_type": "one_time",
                "limit": amount, "expiry_days": None,
                "reason": "Unknown merchant, one-time card created."}
    return {"risk": "low", "action": "no_action", "card_type": None,
            "limit": None, "expiry_days": None,
            "reason": "Trusted merchant, no intervention needed."}


def _extract_decision_json(full_text: str):
    if "DECISION_JSON:" not in full_text:
        return None
    json_part = full_text.split("DECISION_JSON:")[-1].strip()
    return _safe_json_parse(json_part, fallback=None)


@app.post("/analyze")
async def analyze(ctx: CheckoutContext):
    """Legacy text-based streaming analyze endpoint."""
    user_message = (
        f"Checkout context:\n"
        f"- Merchant: {ctx.merchant_name}\n"
        f"- Scenario: {ctx.scenario}\n"
        f"- Amount: EUR {ctx.amount:.2f}\n"
        f"- Description: {ctx.description}\n\n"
        f"Analyse this and tell me how to protect this payment."
    )

    def stream_claude():
        full_text = ""
        try:
            with ai.messages.stream(
                model=MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    yield _sse({"type": "text", "content": text})

            decision = _extract_decision_json(full_text)
            if not decision:
                decision = _fallback_decision(ctx.scenario, ctx.amount)
            yield _sse({"type": "decision", "content": decision})
            yield _sse({"type": "done"})
        except Exception as e:
            yield _sse({"type": "error", "content": str(e)})

    return StreamingResponse(
        stream_claude(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Agentic image analysis - this is the primary endpoint the frontend uses.
@app.post("/analyze-image-upload")
async def analyze_image_upload(
    file: UploadFile = File(...),
    hint: str | None = Form(None),
):
    """Runs the agentic ShieldPay loop on an uploaded image."""
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    media_type = file.content_type or "image/jpeg"

    return StreamingResponse(
        run_shieldpay_agent(image_b64, media_type, hint),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/create-card")
def create_card(req: CreateCardRequest):
    """Direct card creation endpoint, kept for non-agent flows."""
    result = _create_bunq_card(req.scenario, req.limit, req.expiry_days)
    if not result.get("created"):
        raise HTTPException(status_code=502,
                            detail=result.get("error", "Card creation failed"))
    return {
        "card_id": result["card_id"],
        "second_line": result["second_line"],
        "limit": result["limit"],
        "currency": result["currency"],
        "scenario": result["scenario"],
        "expiry_date": result["expiry_date"],
        "status": result["status"],
        "masked_number": result["masked_number"],
        "created_at": datetime.utcnow().isoformat(),
    }


@app.post("/cancel-card")
def cancel_card(req: CancelCardRequest):
    cancel_body = json.dumps({"status": "CANCELLED", "cancellation_reason": "NONE"})
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card/{req.card_id}"
    resp = requests.put(url, headers=bunq_headers(cancel_body), data=cancel_body)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"bunq cancel error: {resp.text}")
    return {
        "card_id": req.card_id,
        "status": "CANCELLED",
        "cancelled_at": datetime.utcnow().isoformat(),
    }


@app.get("/get-card/{card_id}")
def get_card(card_id: int):
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card/{card_id}"
    resp = requests.get(url, headers=bunq_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"bunq error: {resp.text}")

    card_data = _extract_card_data(resp.json())
    if not card_data:
        raise HTTPException(status_code=502,
                            detail="Could not parse card from bunq response")
    return _format_card_for_display(card_data)


@app.get("/default-card")
def default_card():
    """Returns the user's default virtual card. Creates one if missing.

    Sets created_now=True on the response that performed the creation, so the
    UI can surface a notice. Cached for the lifetime of the process.
    """
    return _get_or_create_default_card()


def _extract_card_data(bunq_response: dict):
    for item in bunq_response.get("Response", []):
        if "CardDebit" in item:
            return item["CardDebit"]
        if "Card" in item:
            return item["Card"]
    return None


def _format_card_for_display(card: dict) -> dict:
    limit = card.get("card_limit") or {}
    return {
        "card_id": card.get("id"),
        "name_on_card": card.get("name_on_card"),
        "second_line": card.get("second_line"),
        "expiry_date": card.get("expiry_date"),
        "status": card.get("status"),
        "limit_value": limit.get("value"),
        "limit_currency": limit.get("currency", "EUR"),
        "type": card.get("type"),
        "product_type": card.get("product_type"),
    }


@app.get("/allowed-card-names")
def allowed_card_names():
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card-name"
    resp = requests.get(url, headers=bunq_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"bunq error: {resp.text}")
    for item in resp.json().get("Response", []):
        if "CardUserNameArray" in item:
            return {"allowed_names": item["CardUserNameArray"].get("possible_card_name_array", [])}
    return {"allowed_names": []}


app.mount("/app", StaticFiles(directory="static", html=True), name="static")