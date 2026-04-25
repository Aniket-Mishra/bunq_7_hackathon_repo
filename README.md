# ShieldPay

AI payment protection powered by Claude and bunq.

Upload a checkout screenshot. Claude analyses it, checks the merchant, and
creates a disposable bunq virtual card with a tight spending limit. When the
payment is done, the card is cancelled automatically.

---

## How it works

1. You upload a checkout screenshot.
2. Claude runs an agentic loop: it reads the image, checks the merchant reputation, and reviews your transaction history.
3. If the merchant is risky or unknown, a bunq virtual card is created with an exact or near-zero limit.
4. You pay with that card. Then you click the button and the card is cancelled.

---

## Prerequisites

- Python 3.11+
- A bunq sandbox account
- An Anthropic API key

---

## bunq Setup

You need four things from bunq: a session token, your user ID, your monetary
account ID, and an RSA private key. Follow these steps.

**1. Create a sandbox user**

Go to https://www.bunq.com/sandbox and create a sandbox account.

**2. Install the bunq SDK and run the setup script**

The official bunq Python SDK handles installation and key registration.
Run it once to register your RSA key pair and open a session.

```bash
pip install bunq-sdk-python
```

After running the SDK setup, you will find `installation.key` in the directory
you ran it from. Note down the session token, user ID, and monetary account ID
that the SDK prints out. You will need them in your `.env`.

Alternatively, you can call the bunq API directly:

```
POST https://public-api.sandbox.bunq.com/v1/installation
POST https://public-api.sandbox.bunq.com/v1/device-server
POST https://public-api.sandbox.bunq.com/v1/session-server
```

See https://doc.bunq.com for the full walkthrough.

---

## Installation

```bash
git clone <your-repo>
cd shieldpay

pip install fastapi uvicorn anthropic python-multipart cryptography python-dotenv requests
```

---

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...

SESSION_TOKEN=<from bunq session-server response>
USER_ID=<your bunq user ID>
MONETARY_ACCOUNT_ID=<your bunq monetary account ID>

BUNQ_INSTALLATION_KEY_PATH=./bunq-user-1/installation.key
BUNQ_API_URL=https://public-api.sandbox.bunq.com/v1

CARD_NAME=Card Holder
```

`CARD_NAME` must match one of the allowed names on your bunq account. You can
check which names are allowed by hitting `GET /allowed-card-names` once the
server is running.

---

## Project Structure

```
shieldpay/
  main.py          # FastAPI backend and agent logic
  static/
    index.html     # Frontend (single file)
  bunq-user-1/
    installation.key
  .env
```

---

## Running

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

---

## API Endpoints

| Method | Path                   | Description                              |
|--------|------------------------|------------------------------------------|
| POST   | /analyze-image-upload  | Main endpoint. Runs the agentic loop.    |
| POST   | /cancel-card           | Cancels a bunq card by ID.               |
| GET    | /default-card          | Gets or creates the user's default card. |
| GET    | /get-card/{card_id}    | Fetches card details from bunq.          |
| POST   | /create-card           | Creates a card directly, no agent.       |
| GET    | /allowed-card-names    | Lists bunq-permitted card names.         |
| GET    | /health                | Sanity check.                            |

---

## Notes

- All bunq calls go to the sandbox by default. Change `BUNQ_API_URL` to switch to production.
- Transaction history is mocked. To use real data, replace `MOCK_TRANSACTIONS` with a live call to the bunq payments API.
- The agent uses `claude-sonnet-4-6`. Do not change this to a model that lacks tool use support.