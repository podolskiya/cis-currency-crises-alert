<div align="center">

# Frontier Early Warning System

**Currency crisis probabilities for 18 frontier economies across Central Asia, the Caucasus, Eastern Europe and the Western Balkans**

*A Kaminsky–Reinhart–Végh signal model rebuilt for the post-Soviet space, with regime conditioning and honest out-of-sample validation.*

</div>

---

## The problem

Early warning systems for emerging-market currency crises are a mature literature. Almost none are built for this region specifically — and the ones applied to it inherit assumptions that break here: dollar anchors for euro-pegged Balkan states, floating-rate pressure measures for currency boards, and pooled coefficients across economies that absorb shocks through entirely different valves.

This model treats those differences as the subject rather than the noise.

## What it does

Estimates the probability that a currency crisis **begins within the next 12 months**, using only IMF exchange-rate and international-liquidity data available **three months prior** — the real publication lag, not a hindsight-flattered one.

<table>
<tr><td width="33%" valign="top">

**Crisis dating**

Exchange market pressure index with country-specific inverse-SD weights, expanding and lagged moments, and a reserve-only variant for currency boards whose FX term is degenerate by construction.

</td><td width="33%" valign="top">

**Regime conditioning**

*De facto* peg classification derived from trailing volatility, not hard-coded — so Azerbaijan reads as pegged until February 2015 and floating after, without the model being told.

</td><td width="33%" valign="top">

**Honest validation**

Expanding-window folds with a 12-month purge gap, because a label at month *t* encodes onsets through *t+12* and would otherwise leak into the test period.

</td></tr>
</table>

---

## Crisis chronology

44 episodes detected from 1996 to 2026, generated from the data rather than hand-coded:

| Wave | Episodes |
|---|---|
| **Russian contagion, 1998–99** | Russia (Aug 1998, z=5.19), Ukraine, Moldova, Georgia, Kazakhstan, Azerbaijan |
| **Global financial crisis, 2008–09** | Ukraine, Turkey, Serbia, Bosnia, Armenia, Belarus, Georgia, Kazakhstan, Mongolia, Moldova, Tajikistan |
| **Oil and rouble, 2014–16** | Azerbaijan (Feb 2015, z=10.91), Russia, Belarus, Kazakhstan, Georgia, Moldova, Kyrgyzstan, Armenia, Mongolia, Tajikistan |
| **COVID, 2020** | Kazakhstan, Georgia, Belarus, Kyrgyzstan, Albania, Turkey |
| **War, 2022** | Ukraine, Kazakhstan, Tajikistan |
| **Idiosyncratic** | Belarus 2011 (65% devaluation), Turkey 2001 and 2018, Mongolia 2016 |

> Bosnia's October 2008 episode fires on a 16.4% reserve drawdown with the currency board holding at **exactly 0.0% FX movement** — a pressure event the standard index is structurally incapable of detecting.

---

## Results

Production model: **logistic regression, 22 features** (country fundamentals + regime conditioning). Four scorable folds spanning 2011–2023, 429 positive observations across 4,465 country-months.

| Specification | Mean AUC | SD | Min | Verdict |
|---|:---:|:---:|:---:|---|
| Logistic, country only | 0.637 | 0.121 | 0.522 | Unstable across regimes |
| **Logistic, country + regime** | **0.654** | **0.036** | **0.605** | **Production** |
| Logistic, + real overvaluation | 0.613 | 0.077 | 0.525 | Rejected |
| Logistic, all blocks | 0.600 | 0.079 | 0.500 | Rejected |
| LightGBM, best configuration | 0.572 | 0.100 | 0.433 | Rejected |

Average precision 0.233 against a 0.139 base rate — roughly 1.7× better than chance at ranking.

### What failed, and why that matters

Four theoretically motivated additions were tested on identical folds and **all four underperformed**:

- **Gradient boosting** loses by 0.08 AUC and drops below random on country fundamentals alone. Its top feature by split count is the reserve-scale proxy — with 429 positives across 17 countries, the trees split on *which country* rather than on what is happening to it.
- **Cross-sectional contagion** aggregates add −0.007. The Russia transmission term carries a genuine, sign-stable coefficient; the regional aggregates around it do not.
- **Real overvaluation** costs −0.040 as a block, despite `rer_vs_60m` and `infl_accel` being selected in *every* fold by an honest per-fold selector. The signal is real; six features to extract two is not worth the variance.
- **Bootstrap feature selection** costs −0.054, with only a 0.015 gap between the optimistic and honest versions.

The consistency is the finding: at this sample size, the variance cost of added complexity exceeds its signal. Regime conditioning was the single addition that helped — and it improved the *minimum* fold from 0.522 to 0.605 while cutting dispersion by two thirds, which for an early warning system matters more than the mean.

---

## Limitations

Stated plainly, because a model of this kind is only as useful as its caveats are visible.

- **No crisis onset since July 2022.** The last validated test window ends March 2023. Every reading after that date is extrapolation from a model whose most recent evidence predates it.
- **Probabilities are not calibrated** to observed frequencies. Percentile rank against a country's own history is the more reliable signal, and the dashboard leads with it.
- **Data freshness varies by country** — most lag one to two months, Russia ten, Bosnia fifteen. Turkmenistan stopped reporting exchange rates to the Fund in 2001 and cannot be scored at all.
- **Regularisation spreads signal across correlated features.** No single driver carries a score, so attribution is presented grouped rather than as a top-N list.
- **Research tool, not investment advice.**

---

## Pipeline

```
src/
├── config.py              panel definition, anchors, regimes, resolved IMF codes
├── ingest/
│   ├── imf_er.py          exchange rates, USD and EUR anchors
│   ├── clean_er.py        anchor selection, monthly grid, log differences
│   ├── imf_il.py          international liquidity (reserves ex gold)
│   ├── clean_il.py        merge, drawdown series, freshness audit
│   └── imf_cpi.py         consumer prices, panel plus anchor economies
├── features/
│   ├── emp.py             exchange market pressure index, crisis dating
│   ├── target.py          12-month forward label, post-crisis exclusion
│   ├── contagion.py       leave-one-out cross-sectional features
│   ├── reer.py            real exchange rate, inflation differentials
│   └── build.py           feature assembly, publication lag, regime block
├── model/
│   ├── splits.py          expanding-window CV with purge gap
│   ├── baseline.py        nested logistic specifications
│   ├── model.py           LightGBM comparison
│   └── production.py      final fit, scoring, grouped attribution, JSON export
└── web/public/            static dashboard
```

**Run order**

```bash
python src/ingest/imf_er.py     && python src/ingest/clean_er.py
python src/ingest/imf_il.py     && python src/ingest/clean_il.py
python src/ingest/imf_cpi.py
python src/features/emp.py      && python src/features/target.py
python src/features/contagion.py && python src/features/reer.py
python src/features/build.py
python src/model/production.py
```

---

## Data

IMF SDMX 3.0 API at `api.imf.org/external/sdmx/3.0`, dataflows `ER`, `IL` and `CPI`. No API key required.

> **A note for anyone reusing this.** The IMF retired International Financial Statistics and refactored its dataflows across 2025–26. Series codes from older tutorials no longer resolve, the legacy `dataservices.imf.org` endpoint is being retired, and dataflows now belong to departmental sub-agencies (`IMF.STA`, `IMF.RES`, `IMF.FAD`) rather than a bare `IMF` agency. The discovery scripts in `scripts/` resolve current codes empirically rather than assuming them.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

<div align="center">
<sub>Built with IMF public data · Not affiliated with the IMF · MIT licensed</sub>
</div>
