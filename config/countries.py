COUNTRIES = [
    {"name": "Kazakhstan",      "iso2": "KZ", "iso3": "KAZ"},
    {"name": "Kyrgyz Republic", "iso2": "KG", "iso3": "KGZ"},
    {"name": "Tajikistan",      "iso2": "TJ", "iso3": "TJK"},
    {"name": "Turkmenistan",    "iso2": "TM", "iso3": "TKM"},
    {"name": "Uzbekistan",      "iso2": "UZ", "iso3": "UZB"},
    {"name": "Armenia",         "iso2": "AM", "iso3": "ARM"},
    {"name": "Azerbaijan",      "iso2": "AZ", "iso3": "AZE"},
    {"name": "Georgia",         "iso2": "GE", "iso3": "GEO"},
    {"name": "Russia",          "iso2": "RU", "iso3": "RUS"},
    {"name": "Belarus",         "iso2": "BY", "iso3": "BLR"},
    {"name": "Moldova",         "iso2": "MD", "iso3": "MDA"},
]

ISO2_LIST = [c["iso2"] for c in COUNTRIES]
ISO3_LIST = [c["iso3"] for c in COUNTRIES]
ISO3_TO_NAME = {c["iso3"]: c["name"] for c in COUNTRIES}
ISO2_TO_NAME = {c["iso2"]: c["name"] for c in COUNTRIES}
ISO2_TO_ISO3 = {c["iso2"]: c["iso3"] for c in COUNTRIES}
ISO3_TO_ISO2 = {c["iso3"]: c["iso2"] for c in COUNTRIES}