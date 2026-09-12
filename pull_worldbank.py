"""
Phase 1a — World Bank WDI pull for the Frontier/CIS Early Warning System.

Pulls annual macro fundamentals for the 11-country panel via the World Bank
API (no API key required). Saves one long-format CSV and prints a coverage
summary so we can see exactly what came back before building anything on
top of it.

Run:
    pip install requests pandas
    python pull_worldbank.py
"""

import time
import requests
import pandas as pd

# --- Panel definition -------------------------------------------------

COUNTRIES = {
    "KAZ": "Kazakhstan",
    "KGZ": "Kyrgyz Republic",
    "TJK": "Tajikistan",
    "TKM": "Turkmenistan",
    "UZB": "Uzbekistan",
    "ARM": "Armenia",
    "AZE": "Azerbaijan",
    "GEO": "Georgia",
    "RUS": "Russian Federation",
    "BLR": "Belarus",
    "MDA": "Moldova",
}

# WDI indicator code -> our short name
# NOTE: GC.BAL.CASH.GD.ZS (fiscal balance) deliberately excluded — that
# series is built on a pre-2000s cash-basis GFS standard the World Bank
# has effectively stopped populating (no data in any recent vintage, for
# any country). Fiscal balance will be sourced from IMF WEO in Phase 1c
# instead of chasing a weaker WDI substitute.
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "DT.DOD.DECT.GN.ZS": "external_debt_pct_gni",
    "BX.TRF.PWKR.DT.GD.ZS": "remittances_pct_gdp",
    "TX.VAL.FUEL.ZS.UN": "fuel_exports_pct_merch_exports",
}

START_YEAR, END_YEAR = 1995, 2025
BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
MAX_RETRIES = 3


def fetch_series(country_iso3: str, indicator_code: str) -> list[dict]:
    """Fetch one country/indicator series, handling WDI's pagination and
    retrying transient timeouts before giving up on that series."""
    url = BASE_URL.format(country=country_iso3, indicator=indicator_code)
    params = {
        "format": "json",
        "per_page": 1000,
        "date": f"{START_YEAR}:{END_YEAR}",
    }

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            # WDI returns [metadata, data] on success, or a dict on error
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                return []
            return payload[1]
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)  # 2s, then 4s backoff
    raise last_exc


def main():
    rows = []
    for iso3, country_name in COUNTRIES.items():
        for code, short_name in INDICATORS.items():
            try:
                records = fetch_series(iso3, code)
            except requests.RequestException as exc:
                print(f"  FAILED  {iso3} / {code}: {exc}")
                continue

            for rec in records:
                if rec.get("value") is None:
                    continue
                rows.append(
                    {
                        "country_iso3": iso3,
                        "country_name": country_name,
                        "indicator_code": code,
                        "indicator": short_name,
                        "year": int(rec["date"]),
                        "value": float(rec["value"]),
                    }
                )
            time.sleep(0.2)  # polite pacing, not strictly required by the API

    df = pd.DataFrame(rows)
    df.to_csv("worldbank_raw_cis.csv", index=False)

    print(f"\nSaved {len(df)} observations to worldbank_raw_cis.csv\n")
    print("=== Coverage summary (non-null observations, year range) ===")
    if df.empty:
        print("No data returned at all — check network access / indicator codes.")
        return

    summary = (
        df.groupby(["indicator", "country_iso3"])["year"]
        .agg(n_obs="count", min_year="min", max_year="max")
        .reset_index()
    )
    pivot = summary.pivot(index="country_iso3", columns="indicator", values="n_obs")
    pivot = pivot.reindex(index=COUNTRIES.keys(), columns=INDICATORS.values())
    print(pivot.to_string())


if __name__ == "__main__":
    main()