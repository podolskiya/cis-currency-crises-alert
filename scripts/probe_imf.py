import httpx, sys

CANDIDATES = [
    ("sdmx3-dataflow-json",
     "https://api.imf.org/external/sdmx/3.0/structure/dataflow/IMF",
     {"Accept": "application/vnd.sdmx.structure+json;version=2.0.0"}),
    ("sdmx3-dataflow-xml",
     "https://api.imf.org/external/sdmx/3.0/structure/dataflow/IMF",
     {"Accept": "application/vnd.sdmx.structure+xml;version=3.0.0"}),
    ("datamapper-indicators",
     "https://www.imf.org/external/datamapper/api/v1/indicators",
     {"Accept": "application/json"}),
    ("bis-dataflow",
     "https://stats.bis.org/api/v2/structure/dataflow/BIS",
     {"Accept": "application/vnd.sdmx.structure+xml;version=2.1"}),
]

for name, url, headers in CANDIDATES:
    try:
        r = httpx.get(url, headers=headers, timeout=45, follow_redirects=True)
        body = r.text[:300].replace("\n", " ")
        print(f"[{name}] {r.status_code} | len={len(r.content)} | {body}\n")
    except Exception as e:
        print(f"[{name}] FAILED: {type(e).__name__}: {e}\n")