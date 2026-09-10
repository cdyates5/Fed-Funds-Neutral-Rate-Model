# US Neutral Rate Monitor

Self-refreshing macro dashboard: a six-model composite estimate of the US natural rate
of interest (r\*), the nominal policy corridor against effective fed funds, and a
conditional event study of what has historically followed restrictive policy.

Rebuilt automatically every weekday by GitHub Actions and published to GitHub Pages.

---

## One-time setup

### 1. Create the repo

```bash
git init
git add .
git commit -m "feat: neutral rate monitor"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### 2. Add the FRED API key as a secret

Get a free key at <https://fredaccount.stlouisfed.org/apikeys>, then:

**Settings → Secrets and variables → Actions → New repository secret**

| Field | Value |
|---|---|
| Name | `FRED_API_KEY` |
| Secret | your 32-character key |

The build fails fast with a clear message if this is missing. Never commit the key —
`build.py` reads it only from the environment.

### 3. Enable GitHub Pages

**Settings → Pages → Build and deployment → Source: `GitHub Actions`**

Do *not* pick "Deploy from a branch" — this repo publishes a build artifact.

### 4. Run it once

**Actions → Build and deploy dashboard → Run workflow.**

First run takes roughly 2–3 minutes. The dashboard then lives at
`https://<you>.github.io/<repo>/`.

---

## How it refreshes

The workflow runs **06:15 UTC, Monday–Friday** (`cron: "15 6 * * 1-5"`), plus on manual
dispatch and whenever `build.py` changes. Each run re-fetches everything from source, so
the dashboard tracks the underlying data with no manual step.

Change the cadence by editing the `cron` line in `.github/workflows/build.yml`.
Cron is always UTC, and GitHub delays scheduled jobs under load — treat the time as
approximate. Weekly (`"15 6 * * 1"`) is perfectly adequate given the release calendar
below.

### What actually changes, and when

| Input | Source | Update frequency |
|---|---|---|
| Effective fed funds, core PCE, CPI, payrolls, industrial production, GDP | FRED API | monthly / quarterly, on release |
| TIPS 5y & 10y real yields | FRED (daily) | every business day |
| FOMC longer-run median dot | FRED `FEDTARMDLR` | quarterly, at SEP meetings |
| Holston–Laubach–Williams r\* | NY Fed `.xlsx` | quarterly |
| Lubik–Matthes r\* | Richmond Fed `.xlsx` | quarterly |
| S&P 500 | Yahoo Finance (daily) | every business day |

The two structural r\* models refresh quarterly, so most daily runs move only the market
and stance readings. That is expected, not a fault.

---

## Repo layout

```
build.py                      # fetch -> compute -> emit public/index.html (single script)
requirements.txt
cache/sp500_monthly.csv       # committed fallback; refreshed by CI on every good run
.github/workflows/build.yml   # schedule, build, verify, Pages deploy
build/                        # gitignored intermediates
public/                       # gitignored generated site (index.html, build.json)
```

`public/build.json` is a small machine-readable stamp (`built_utc`, `data_through`,
`composite_rstar`, `stance_bp`) — useful for uptime checks or a status badge.

---

## Running locally

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here      # Windows: set FRED_API_KEY=...
python build.py
open public/index.html
```

---

## Resilience

- **Yahoo rate limits.** GitHub-hosted runners share IP ranges that Yahoo throttles.
  `build.py` retries three times with backoff, then falls back to
  `cache/sp500_monthly.csv`. The workflow commits the refreshed cache after each
  successful fetch, so a Yahoo outage degrades to slightly stale equity data rather
  than a failed build.
- **Output verification.** The workflow asserts `public/index.html` exists and exceeds
  100 KB before deploying, so a truncated build never replaces a good page.
- **Fed source URLs.** The NY Fed and Richmond Fed spreadsheet links are hardcoded. If
  either institution reorganises its site the build fails loudly — fix the URL near the
  top of `build.py`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: FRED_API_KEY is not set` | Add the repository secret (step 2); confirm the name matches exactly |
| Pages 404 after a green run | Set Pages source to **GitHub Actions** (step 3) |
| `Permission denied` pushing the cache | Settings → Actions → General → Workflow permissions → **Read and write** |
| Scheduled runs stopped | GitHub disables schedules on repos with no activity for 60 days — push a commit or click Run workflow |
| Charts blank, page loads | Chart.js is loaded from cdnjs; check the browser console and any network restrictions |

---

## Methodology

Documented in full on the dashboard itself, under each tab's *Methodology* section —
pillar definitions, the composite and stance construction, HAC inference on overlapping
forward windows, and the caveats (endogeneity, revision, sample size).

Sources: FRED (St. Louis Fed) · NY Fed Holston–Laubach–Williams · Richmond Fed
Lubik–Matthes · Yahoo Finance. Not investment advice.
