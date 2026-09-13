"""Panel definition. Single source of truth for country scope and codes."""

COUNTRIES = {
    "KAZ": {"name": "Kazakhstan",      "iso2": "KZ", "ccy": "KZT", "caveat": None},
    "UZB": {"name": "Uzbekistan",      "iso2": "UZ", "ccy": "UZS", "caveat": "pre-2017 dual FX regime"},
    "KGZ": {"name": "Kyrgyz Republic", "iso2": "KG", "ccy": "KGS", "caveat": None},
    "TJK": {"name": "Tajikistan",      "iso2": "TJ", "ccy": "TJS", "caveat": "thin reporting"},
    "TKM": {"name": "Turkmenistan",    "iso2": "TM", "ccy": "TMT", "caveat": "official data unreliable; parallel rate diverges"},
    "GEO": {"name": "Georgia",         "iso2": "GE", "ccy": "GEL", "caveat": None},
    "ARM": {"name": "Armenia",         "iso2": "AM", "ccy": "AMD", "caveat": None},
    "AZE": {"name": "Azerbaijan",      "iso2": "AZ", "ccy": "AZN", "caveat": "managed peg; EMP understates stress"},
    "UKR": {"name": "Ukraine",         "iso2": "UA", "ccy": "UAH", "caveat": "wartime capital controls from 2022"},
    "MDA": {"name": "Moldova",         "iso2": "MD", "ccy": "MDL", "caveat": None},
    "BLR": {"name": "Belarus",         "iso2": "BY", "ccy": "BYN", "caveat": "2016 redenomination; sanctioned"},
    "RUS": {"name": "Russia",          "iso2": "RU", "ccy": "RUB", "caveat": "sanctioned; reserves reporting suspended 2022"},
}

ISO3 = list(COUNTRIES)
SDMX_KEY = "+".join(ISO3)          # e.g. KAZ+UZB+KGZ... for SDMX dimension keys
PANEL_START = "1995-01"

IMF_BASE = "https://api.imf.org/external/sdmx/3.0"

ER = {
    "agency": "IMF.STA",
    "flow": "ER",
    "key_order": ["COUNTRY", "INDICATOR", "TYPE_OF_TRANSFORMATION", "FREQUENCY"],
    "indicator": "XDC_USD",      
    "transformations": ["EOP_RT", "PA_RT"],
    "freq": "M",
}

PANEL_START = "1996-01"
COUNTRY_START_OVERRIDE = {
    "BLR": "2000-02",  
    "UZB": "1999-06",   
    "TJK": "1998-01", 
}
EXCLUDE_FROM_MODEL = ["TKM"]                  
PREFER_PA = ["UZB"]                           
POLICY_REFORM_EVENTS = [("UZB", "2017-09")]   

IL = {
    "agency": "IMF.STA",
    "flow": "IL",
    "key_order": ["COUNTRY", "INDICATOR", "UNIT", "FREQUENCY"],
    "indicators": {
        "RXF11_REVS": "USD",        # reserves ex gold — EMP component
        "RXF11FX_REVS": "USD",      # FX subcomponent — feature
        "RXF11_REVS_WIMP": "_Z",    # weeks of imports — feature (unit may differ)
    },
    "freq": "M",
}