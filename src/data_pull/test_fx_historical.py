
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CURRENCY_CODES = {
    "Kazakhstan": "KZT", "Kyrgyz Republic": "KGS", "Tajikistan": "TJS",
    "Turkmenistan": "TMT", "Uzbekistan": "UZS", "Armenia": "AMD",
    "Azerbaijan": "AZN", "Georgia": "GEL", "Russia": "RUB",
    "Belarus": "BYN", "Moldova": "MDL",
}

BASE_URL = "https://api.currencybeacon.com/v1/historical"

TEST_DATES = ["1999-12-31", "2010-06-30", "2022-03-01", "2024-01-31"]


def main():
    api_key = os.getenv("CURRENCYBEACON_API_KEY")
    if not api_key:
        print("No CURRENCYBEACON_API_KEY found in .env")
        return

    for date in TEST_DATES:
        params = {
            "api_key": api_key,
            "date": date,
            "base": "USD",
            "symbols": ",".join(CURRENCY_CODES.values()),
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        print(f"\n--- {date} --- status {resp.status_code}")
        print(resp.text[:500])
        time.sleep(1)


if __name__ == "__main__":
    main()