COUNTRIES = {
    "KAZ": {"name": "Kazakhstan",      "ccy": "KZT", "anchor": "USD", "regime": "managed",
            "caveat": None},
    "UZB": {"name": "Uzbekistan",      "ccy": "UZS", "anchor": "USD", "regime": "managed",
            "caveat": "2017 FX liberalisation; pre-1999 reserves unusable"},
    "KGZ": {"name": "Kyrgyz Republic", "ccy": "KGS", "anchor": "USD", "regime": "managed",
            "caveat": None},
    "TJK": {"name": "Tajikistan",      "ccy": "TJS", "anchor": "USD", "regime": "managed",
            "caveat": "small reserve base; reporting breaks to 2017"},
    "TKM": {"name": "Turkmenistan",    "ccy": "TMT", "anchor": "USD", "regime": "peg",
            "caveat": "no IMF data after 2001"},
    "GEO": {"name": "Georgia",         "ccy": "GEL", "anchor": "USD", "regime": "float",
            "caveat": None},
    "ARM": {"name": "Armenia",         "ccy": "AMD", "anchor": "USD", "regime": "managed",
            "caveat": None},
    "AZE": {"name": "Azerbaijan",      "ccy": "AZN", "anchor": "USD", "regime": "managed",
            "caveat": "peg broke 2015; pressure hides in reserves"},
    "UKR": {"name": "Ukraine",         "ccy": "UAH", "anchor": "USD", "regime": "managed",
            "caveat": "wartime capital controls from 2022"},
    "MDA": {"name": "Moldova",         "ccy": "MDL", "anchor": "EUR", "regime": "managed",
            "caveat": None},
    "BLR": {"name": "Belarus",         "ccy": "BYN", "anchor": "USD", "regime": "managed",
            "caveat": "2000 redenomination; sanctioned"},
    "RUS": {"name": "Russia",          "ccy": "RUB", "anchor": "USD", "regime": "managed",
            "caveat": "sanctioned; reserves reporting lags ~10 months"},
    "SRB": {"name": "Serbia",          "ccy": "RSD", "anchor": "EUR", "regime": "managed",
            "caveat": None},
    "ALB": {"name": "Albania",         "ccy": "ALL", "anchor": "EUR", "regime": "managed",
            "caveat": None},
    "MKD": {"name": "North Macedonia", "ccy": "MKD", "anchor": "EUR", "regime": "peg",
            "caveat": "de facto EUR peg; reserve-only EMP"},
    "BIH": {"name": "Bosnia and Herzegovina", "ccy": "BAM", "anchor": "EUR", "regime": "peg",
            "caveat": "currency board since 1997; FX term degenerate"},
    "MNG": {"name": "Mongolia",        "ccy": "MNT", "anchor": "USD", "regime": "managed",
            "caveat": None},
    "TUR": {"name": "Turkiye",         "ccy": "TRY", "anchor": "USD", "regime": "float",
            "caveat": "2018 and 2021-23 crises; large EM, not frontier"},
}

ISO3 = list(COUNTRIES)
SDMX_KEY = "+".join(ISO3)

USD_ANCHORED = [c for c, v in COUNTRIES.items() if v["anchor"] == "USD"]
EUR_ANCHORED = [c for c, v in COUNTRIES.items() if v["anchor"] == "EUR"]
PEGGED = [c for c, v in COUNTRIES.items() if v["regime"] == "peg"]

PANEL_START = "1996-01"
COUNTRY_START_OVERRIDE = {
    "BLR": "2000-02",   # 1000:1 redenomination and rate unification #
    "UZB": "1999-06",   # reserve series unusable before this #
    "TJK": "1998-01",   # sub-$50m reserve base, unique reporting #
    "SRB": "2006-01",   # reserves series begins here #
    "BIH": "1998-01",   # post-currency-board stabilisation #
}

EXCLUDE_FROM_MODEL = ["TKM"]         
PREFER_PA = ["UZB"]                   # EOP 56% missing #
POLICY_REFORM_EVENTS = [("UZB", "2017-09")]   # FX liberalisation #

IMF_BASE = "https://api.imf.org/external/sdmx/3.0"

ER = {
    "agency": "IMF.STA",
    "flow": "ER",
    "key_order": ["COUNTRY", "INDICATOR", "TYPE_OF_TRANSFORMATION", "FREQUENCY"],
    "indicator_usd": "XDC_USD",
    "indicator_eur": "XDC_EUR",   
    "transformations": ["EOP_RT", "PA_RT"],
    "freq": "M",
}

IL = {
    "agency": "IMF.STA",
    "flow": "IL",
    "key_order": ["COUNTRY", "INDICATOR", "UNIT", "FREQUENCY"],
    "indicators": {
        "RXF11_REVS": "USD",   
        "RXF11FX_REVS": "USD",   
    },
    "freq": "M",
}

PEG_VOL_THRESHOLD = 0.010  
RESERVE_ONLY_EMP = ["BIH", "MKD"]   # FX term degenerate; pressure appears in reserves #