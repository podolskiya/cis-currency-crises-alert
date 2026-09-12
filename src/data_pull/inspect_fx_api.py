"""
Daily FX overlay - step 1 of 2 (inspection).

Confirms API access, response format, and real currency coverage
before writing the full historical puller. CurrencyBeacon is used
because it's one of the few free-tier FX providers with daily rates
for thin/exotic currencies (KZT, AMD, GEL, AZN, etc) -- most free FX
APIs only cover major currency pairs and would return nothing for
most of our panel.

You'll need a free API key: register at https://currencybeacon.com
Put it in a .env file at the project root (copy .env.example to .env
and fill in your real key there -- .env is gitignored, so it never
gets committed).

Run:
    pip install requests python-dotenv
    python inspect_fx_api.py
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the project root, however deep this script sits
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Currency codes for our 11-country panel
CURRENCY_CODES = {
    "Kazakhstan": "KZT", "Kyrgyz Republic": "KGS", "Tajikistan": "TJS",
    "Turkmenistan": "TMT", "Uzbekistan": "UZS", "Armenia": "AMD",
    "Azerbaijan": "AZN", "Georgia": "GEL", "Russia": "RUB",
    "Belarus": "BYN", "Moldova": "MDL",
}

BASE_URL = "https://api.currencybeacon.com/v1/latest"


def main():
    api_key = os.getenv("CURRENCYBEACON_API_KEY")
    if not api_key:
        api_key = input("No key found in .env -- paste it here for this run only: ").strip()
    if not api_key:
        print("No key available.")
        return

    params = {
        "api_key": api_key,
        "base": "USD",
        "symbols": ",".join(CURRENCY_CODES.values()),
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Raw response:\n{resp.text}")

    if resp.status_code == 200:
        data = resp.json()
        rates = data.get("response", {}).get("rates", {})
        print("\n=== Coverage check against our 11-country panel ===")
        for country, code in CURRENCY_CODES.items():
            status = f"{rates[code]}" if code in rates else "MISSING"
            print(f"  {country:16s} ({code}): {status}")


if __name__ == "__main__":
    main()