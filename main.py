"""
ShieldPay — FastAPI Backend
Run from project root: uvicorn main:app --reload --port 8000

Endpoints:
  POST /analyze      — Claude analyses checkout, returns risk decision (streaming)
  POST /create-card  — Creates a bunq virtual card based on Claude's decision
  POST /cancel-card  — Cancels a bunq card (called after "payment")
  GET  /health       — Sanity check
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
CARD_NAME            = os.getenv("CARD_NAME", "Card Holder")  # must match sandbox user name

# Startup validation 
missing = [k for k, v in {
    "SESSION_TOKEN": SESSION_TOKEN,
    "USER_ID": USER_ID,
    "MONETARY_ACCOUNT_ID": MONETARY_ACCOUNT_ID,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
}.items() if not v]
if missing:
    raise RuntimeError(f"Missing .env variables: {', '.join(missing)}")

# RSA signing ─
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

# FastAPI app ─
app = FastAPI(title="ShieldPay API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response models ─
class CheckoutContext(BaseModel):
    merchant_name: str        # e.g. "Netflix", "gadgets4u.nl", "Amazon"
    scenario: str             # "free_trial" | "unknown_merchant" | "trusted_merchant"
    amount: float             # purchase amount in EUR
    description: str          # human-readable context, e.g. "30-day free trial subscription"

class CreateCardRequest(BaseModel):
    scenario: str
    amount: float
    limit: float
    expiry_days: int | None = None   # None = no special expiry logic

class CancelCardRequest(BaseModel):
    card_id: int

class CheckoutImageContext(BaseModel):
    image_base64: str         # base64-encoded image, no data URI prefix
    media_type: str = "image/jpeg"   # "image/jpeg" | "image/png" | "image/webp" | "image/gif"
    hint: str | None = None   # optional user note, e.g. "this is a subscription signup"

# SYSTEM PROMPT for Claude 
SYSTEM_PROMPT = """You are ShieldPay, an AI payment protection agent. 
Your job is to analyse online checkout situations and decide the safest way to pay.

You will receive a checkout context and must respond with:
1. A brief real-time analysis (think out loud, 2-4 sentences, conversational)
2. Then a JSON block with your decision

Scenarios you handle:
- free_trial: Merchant is offering a free trial that auto-charges later
- unknown_merchant: Merchant is unfamiliar or untrusted
- trusted_merchant: Well-known, reputable merchant with small purchase

Your decisions:
- free_trial → create a virtual card with €0.01 limit (can't be charged after trial)
- unknown_merchant → create a one-time virtual card capped at exact purchase amount
- trusted_merchant → no card needed, payment is safe

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

Be concise, confident, and slightly dramatic — you are protecting someone's money in real time."""

IMAGE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

When analysing an image:
- First identify the merchant name and total amount from the image.
- Infer the scenario yourself: free_trial, unknown_merchant, or trusted_merchant.
- If the image is unreadable or not a bill/receipt, set risk to "low", action to "no_action", and explain in the reason field.
- Include the extracted merchant and amount in your spoken analysis so the user can verify."""

@app.get("/")
def root():
    return {"status": "ShieldPay API running", "docs": "/docs", "health": "/health"}

# Endpoint: health 
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# Endpoint: analyze (streaming) 
@app.post("/analyze")
async def analyze(ctx: CheckoutContext):
    """
    Streams Claude's reasoning + decision back as Server-Sent Events.
    Frontend listens to this stream and renders it in real time.
    
    SSE format:
      data: <text chunk>         ← reasoning text, streamed word by word
      data: [DECISION] {...}     ← final JSON decision object
      data: [DONE]               ← stream complete
    """
    user_message = (
        f"Checkout context:\n"
        f"- Merchant: {ctx.merchant_name}\n"
        f"- Scenario: {ctx.scenario}\n"
        f"- Amount: €{ctx.amount:.2f}\n"
        f"- Description: {ctx.description}\n\n"
        f"Analyse this and tell me how to protect this payment."
    )

    def stream_claude():
        full_text = ""
        try:
            with ai.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    # Stream raw text chunks to frontend
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

            # After streaming, extract the JSON decision
            decision = None
            if "DECISION_JSON:" in full_text:
                json_part = full_text.split("DECISION_JSON:")[-1].strip()
                # Find the JSON object
                start = json_part.find("{")
                end = json_part.rfind("}") + 1
                if start != -1 and end > start:
                    try:
                        decision = json.loads(json_part[start:end])
                    except json.JSONDecodeError:
                        pass

            if decision:
                yield f"data: {json.dumps({'type': 'decision', 'content': decision})}\n\n"
            else:
                # Fallback decision based on scenario if Claude didn't format correctly
                fallback = _fallback_decision(ctx.scenario, ctx.amount)
                yield f"data: {json.dumps({'type': 'decision', 'content': fallback})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        stream_claude(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if deployed
        }
    )

def _fallback_decision(scenario: str, amount: float) -> dict:
    """Fallback if Claude's response doesn't parse cleanly."""
    if scenario == "free_trial":
        return {"risk": "high", "action": "create_card", "card_type": "trial_shield",
                "limit": 0.01, "expiry_days": 29, "reason": "Free trial detected — shield card created."}
    elif scenario == "unknown_merchant":
        return {"risk": "medium", "action": "create_card", "card_type": "one_time",
                "limit": amount, "expiry_days": None, "reason": "Unknown merchant — one-time card created."}
    else:
        return {"risk": "low", "action": "no_action", "card_type": None,
                "limit": None, "expiry_days": None, "reason": "Trusted merchant — no intervention needed."}

# Endpoint: create-card ─
@app.post("/create-card")
def create_card(req: CreateCardRequest):
    """
    Creates a bunq virtual card based on Claude's decision.
    Returns card details to display in the frontend.
    """
    # Determine label based on scenario
    labels = {
        "free_trial":        "TRIAL SHIELD",
        "unknown_merchant":  "ONE-TIME CARD",
        "trusted_merchant":  "SHIELD CARD",
    }
    second_line = labels.get(req.scenario, "SHIELD CARD")

    # Build card creation body
    card_body = {
        "second_line": second_line,
        "name_on_card": CARD_NAME,
        "type": "MASTERCARD",
        "product_type": "MASTERCARD_DEBIT",
        "pin_code_assignment": [
            {
                "type": "PRIMARY",
                # "pin_code": os.getenv("CARD_PIN", "473829"),
                "pin_code": os.getenv("CARD_PIN", "1234"),
                "monetary_account_id": int(MONETARY_ACCOUNT_ID)
            }
        ]
    }

    body_str = json.dumps(card_body)
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card-debit"

    resp = requests.post(url, headers=bunq_headers(body_str), data=body_str)

    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"bunq error: {resp.text}")

    # Parse card ID from response
    response_list = resp.json().get("Response", [])
    card_id = None
    card_data = {}
    for item in response_list:
        if "CardDebit" in item:
            card_data = item["CardDebit"]
            card_id = card_data.get("id")
            break
        if "Id" in item:
            card_id = item["Id"]["id"]

    if not card_id:
        raise HTTPException(status_code=502, detail="Could not parse card ID from bunq response")

    # Set spending limit
    limit_body = json.dumps({
        "card_limit": {"value": f"{req.limit:.2f}", "currency": "EUR"},
        "status": "ACTIVE"
    })
    put_url = f"{BUNQ_API_URL}/user/{USER_ID}/card/{card_id}"
    limit_resp = requests.put(put_url, headers=bunq_headers(limit_body), data=limit_body)

    if limit_resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"bunq limit error: {limit_resp.text}"
        )

    # Calculate display expiry
    if req.expiry_days:
        display_expiry = (datetime.utcnow() + timedelta(days=req.expiry_days)).strftime("%Y-%m-%d")
    else:
        display_expiry = card_data.get("expiry_date", "2030-05-31")

    return {
        "card_id": card_id,
        "second_line": second_line,
        "limit": req.limit,
        "currency": "EUR",
        "scenario": req.scenario,
        "expiry_date": display_expiry,
        "status": "ACTIVE",
        "masked_number": f"**** **** **** {str(card_id)[-4:].zfill(4)}",
        "created_at": datetime.utcnow().isoformat(),
    }

# Endpoint: cancel-card ─
@app.post("/cancel-card")
def cancel_card(req: CancelCardRequest):
    """
    Cancels a bunq card. Called immediately after the demo "payment".
    """
    cancel_body = json.dumps({
        "status": "CANCELLED",
        "cancellation_reason": "NONE",  
    })
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card/{req.card_id}"
    resp = requests.put(url, headers=bunq_headers(cancel_body), data=cancel_body)

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"bunq cancel error: {resp.text}")

    return {
        "card_id": req.card_id,
        "status": "CANCELLED",
        "cancelled_at": datetime.utcnow().isoformat(),
    }


@app.post("/analyze-image")
async def analyze_image(ctx: CheckoutImageContext):
    """
    Same SSE contract as /analyze, but Claude extracts merchant + amount
    from an uploaded bill/receipt image instead of receiving them as text.
    """
    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": ctx.media_type,
                "data": ctx.image_base64,
            },
        },
        {
            "type": "text",
            "text": (
                f"Here is a bill or receipt. Extract the merchant and total amount, "
                f"then analyse the payment risk.\n"
                f"User hint: {ctx.hint or 'none'}"
            ),
        },
    ]

    def stream_claude():
        full_text = ""
        try:
            with ai.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=800,
                system=IMAGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

            decision = None
            if "DECISION_JSON:" in full_text:
                json_part = full_text.split("DECISION_JSON:")[-1].strip()
                start = json_part.find("{")
                end = json_part.rfind("}") + 1
                if start != -1 and end > start:
                    try:
                        decision = json.loads(json_part[start:end])
                    except json.JSONDecodeError:
                        pass

            if decision:
                yield f"data: {json.dumps({'type': 'decision', 'content': decision})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Could not parse decision from image analysis.'})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        stream_claude(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/analyze-image-upload")
async def analyze_image_upload(
    file: UploadFile = File(...),
    hint: str | None = Form(None),
):
    """
    Accepts a real image file upload (multipart/form-data).
    Internally reuses /analyze-image logic.
    """
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    media_type = file.content_type or "image/jpeg"

    ctx = CheckoutImageContext(
        image_base64=image_b64,
        media_type=media_type,
        hint=hint,
    )
    return await analyze_image(ctx)

@app.get("/get-card/{card_id}")
def get_card(card_id: int):
    """
    Fetches real card details from bunq.
    Used by the frontend to display the card after creation.
    Returns only safe-to-display fields (no PAN, no CVV - bunq does not expose these).
    """
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card/{card_id}"
    resp = requests.get(url, headers=bunq_headers())

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"bunq error: {resp.text}")

    card_data = _extract_card_data(resp.json())
    if not card_data:
        raise HTTPException(status_code=502, detail="Could not parse card from bunq response")

    return _format_card_for_display(card_data)


def _extract_card_data(bunq_response: dict) -> dict | None:
    """Pulls the card object out of bunq's nested Response array."""
    response_list = bunq_response.get("Response", [])
    for item in response_list:
        if "CardDebit" in item:
            return item["CardDebit"]
        if "Card" in item:
            return item["Card"]
    return None


def _format_card_for_display(card: dict) -> dict:
    """Shapes bunq's card object into what the frontend needs."""
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
    """
    Returns the list of card names bunq allows for this user.
    Useful for the frontend to show a dropdown if CARD_NAME is not set.
    """
    url = f"{BUNQ_API_URL}/user/{USER_ID}/card-name"
    resp = requests.get(url, headers=bunq_headers())

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"bunq error: {resp.text}")

    response_list = resp.json().get("Response", [])
    for item in response_list:
        if "CardUserNameArray" in item:
            return {"allowed_names": item["CardUserNameArray"].get("possible_card_name_array", [])}

    return {"allowed_names": []}


app.mount("/app", StaticFiles(directory="static", html=True), name="static")