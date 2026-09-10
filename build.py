#!/usr/bin/env python3
"""US Neutral Rate Monitor — CI build script.

Fetches FRED + NY Fed (HLW) + Richmond Fed (Lubik-Matthes) + S&P 500, recomputes the
six-pillar composite r*, the policy stance and the forward-outcome event study, and
writes a self-contained dashboard to public/index.html.

Requires the FRED_API_KEY environment variable (GitHub Actions: repository secret).
Free key: https://fredaccount.stlouisfed.org/apikeys
"""
import os, re, json, time, sys, datetime as dt
import requests, pandas as pd, numpy as np

ROOT     = os.path.dirname(os.path.abspath(__file__))
WORK     = os.path.join(ROOT, "build")     # intermediates (gitignored)
PUBLIC   = os.path.join(ROOT, "public")    # published site
CACHE    = os.path.join(ROOT, "cache")     # committed fallback data
OUT_HTML = os.path.join(PUBLIC, "index.html")
SP_CACHE = os.path.join(CACHE, "sp500_monthly.csv")
for _d in (WORK, PUBLIC, CACHE):
    os.makedirs(_d, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()
if not FRED_API_KEY:
    sys.exit("ERROR: FRED_API_KEY is not set. Add it as a repository secret "
             "(Settings -> Secrets and variables -> Actions) or export it locally.")


def fetch_sp500_monthly(ua):
    """Yahoo daily -> month-end series, with retries and a committed CSV fallback.

    GitHub-hosted runners are periodically rate-limited by Yahoo, so a failed fetch
    falls back to the last good cached copy rather than failing the whole build.
    """
    p1 = int(time.mktime(time.strptime("1962-01-01", "%Y-%m-%d")))
    p2 = int(time.time())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
           f"?period1={p1}&period2={p2}&interval=1d")
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=ua, timeout=90)
            r.raise_for_status()
            j = r.json()["chart"]["result"][0]
            s = pd.Series(j["indicators"]["quote"][0]["close"],
                          index=pd.to_datetime(j["timestamp"], unit="s")).dropna()
            sp_m = s.resample("MS").last()
            if len(sp_m) < 600:
                raise ValueError(f"suspiciously short series ({len(sp_m)} months)")
            sp_m.to_csv(SP_CACHE, header=["close"])
            print(f"S&P 500: fetched {len(sp_m)} months from Yahoo "
                  f"({sp_m.index[-1].date()}), cache refreshed")
            return sp_m
        except Exception as e:
            print(f"  Yahoo attempt {attempt}/3 failed: {e}")
            time.sleep(5 * attempt)
    if os.path.exists(SP_CACHE):
        c = pd.read_csv(SP_CACHE, index_col=0, parse_dates=True)["close"]
        print(f"S&P 500: WARNING - Yahoo unavailable, using cache "
              f"({len(c)} months, through {c.index[-1].date()})")
        return c
    sys.exit("ERROR: S&P 500 fetch failed and no cache exists at " + SP_CACHE)





# ========================= FETCH FRED =========================
import requests, json, pandas as pd, sys

KEY = FRED_API_KEY
BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "FEDFUNDS":   "Effective fed funds, monthly avg",
    "PCEPILFE":   "Core PCE price index",
    "GDPPOT":     "Real potential GDP (CBO), quarterly",
    "DFII5":      "5y TIPS CMT, daily",
    "DFII10":     "10y TIPS CMT, daily",
    "THREEFYTP10":"Kim-Wright 10y term premium, daily",
    "FEDTARMDLR": "FOMC SEP longer-run median dot",
    "USREC":      "NBER recession indicator",
}

out = {}
for sid, desc in SERIES.items():
    r = requests.get(BASE, params={"series_id": sid, "api_key": KEY,
                                   "file_type": "json", "observation_start": "1954-01-01"},
                     timeout=60)
    try:
        obs = r.json()["observations"]
    except Exception:
        print(f"{sid}: FAILED HTTP {r.status_code} {r.text[:150]}")
        continue
    s = pd.Series({o["date"]: (float(o["value"]) if o["value"] != "." else None) for o in obs})
    s.index = pd.to_datetime(s.index)
    s = s.dropna()
    out[sid] = s
    print(f"{sid:12s} {len(s):6d} obs  {s.index[0].date()} -> {s.index[-1].date()}  last={s.iloc[-1]:.3f}  ({desc})")

pd.to_pickle(out, f"{WORK}/fred_raw.pkl")
print("saved", len(out), "series")


# ================= download HLW (NY Fed) and Lubik-Matthes (Richmond Fed) =================
_UA = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
for _fn, _url in {
  f"{WORK}/hlw.xlsx":"https://www.newyorkfed.org/medialibrary/media/research/economists/williams/data/Holston_Laubach_Williams_current_estimates.xlsx",
  f"{WORK}/lm.xlsx":"https://www.richmondfed.org/-/media/RichmondFedOrg/research/economists/bios/data/lubik_matthes_natural_rate_interest.xlsx",
}.items():
    open(_fn, "wb").write(requests.get(_url, headers=_UA, timeout=90).content)
print("downloaded HLW + LM")


# ========================= COMPUTE r* PILLARS =========================
import pandas as pd, numpy as np, json, datetime as dt

raw = pd.read_pickle(f"{WORK}/fred_raw.pkl")

# ---------- monthly master index ----------
START, = ["1965-01-01"]
last_ff = raw["FEDFUNDS"].index[-1]
idx = pd.date_range(START, last_ff, freq="MS")

def to_monthly_mean(s):
    return s.resample("MS").mean()

ff      = raw["FEDFUNDS"].resample("MS").mean()
corepce = raw["PCEPILFE"].resample("MS").last()
core_yoy = (corepce.pct_change(12) * 100.0)

# real fed funds (ex-post, core PCE deflator). ffill core one month so June FF usable
core_yoy_x = core_yoy.reindex(pd.date_range(core_yoy.index[0], last_ff, freq="MS")).ffill(limit=1)
real_ff = (ff - core_yoy_x).dropna()

# ---------- Pillar 1: Kalman local-level trend of realized real FF ----------
def local_level_filter(y, snr=1e-4):
    """One-sided Kalman filter, local level, fixed signal-to-noise ratio."""
    y = y.values.astype(float)
    R = np.nanvar(np.diff(y))          # obs noise scale from first differences
    Q = snr * R
    n = len(y)
    x = np.nanmean(y[:24]); P = R      # diffuse-ish init on first 2y mean
    out = np.empty(n)
    for t in range(n):
        P = P + Q
        K = P / (P + R)
        x = x + K * (y[t] - x)
        P = (1 - K) * P
        out[t] = x
    return out

trend = pd.Series(local_level_filter(real_ff, snr=1e-4), index=real_ff.index)

# ---------- Pillar 2: potential growth anchor (CBO GDPPOT yoy) ----------
gpot = raw["GDPPOT"]
gpot = gpot[gpot.index <= last_ff]                       # drop CBO projection tail
g_yoy = (gpot.pct_change(4) * 100.0).dropna()
g_m = g_yoy.resample("MS").interpolate("linear").reindex(
        pd.date_range(g_yoy.index[0], last_ff, freq="MS")).interpolate("linear").ffill()

# ---------- Pillar 3: market-implied 5y5y forward real (TIPS) ----------
f5y5y = (2.0 * to_monthly_mean(raw["DFII10"]) - to_monthly_mean(raw["DFII5"])).dropna()

# ---------- Pillar 4: FOMC SEP longer-run median dot minus 2% target ----------
sep = (raw["FEDTARMDLR"] - 2.0)
sep_m = sep.resample("MS").last().reindex(pd.date_range(sep.index[0].replace(day=1), last_ff, freq="MS")).ffill()

# ---------- Pillar 5: HLW (NY Fed, quarterly one-sided) ----------
hlw_df = pd.read_excel(f"{WORK}/hlw.xlsx", sheet_name="HLW Estimates", header=None)
dates = pd.to_datetime(hlw_df.iloc[6:, 0])
hlw = pd.Series(hlw_df.iloc[6:, 10].astype(float).values, index=dates).dropna()
hlw_m = hlw.resample("MS").interpolate("linear").reindex(
        pd.date_range(hlw.index[0], last_ff, freq="MS")).interpolate("linear").ffill()

# ---------- Pillar 6: Lubik-Matthes (Richmond Fed, quarterly) ----------
lm_df = pd.read_excel(f"{WORK}/lm.xlsx", sheet_name="LM_RealRate", header=0)
lm = pd.Series(lm_df["Median"].values, index=pd.to_datetime(lm_df["Date"])).dropna()
lm_lo = pd.Series(lm_df["Lower Bound"].values, index=pd.to_datetime(lm_df["Date"])).dropna()
lm_hi = pd.Series(lm_df["Upper Bound"].values, index=pd.to_datetime(lm_df["Date"])).dropna()
lm_m = lm.resample("MS").interpolate("linear").reindex(
        pd.date_range(lm.index[0], last_ff, freq="MS")).interpolate("linear").ffill()

# ---------- assemble ----------
P = pd.DataFrame({
    "trend": trend, "g": g_m, "mkt": f5y5y, "sep": sep_m, "hlw": hlw_m, "lm": lm_m
}).reindex(idx)

P["composite"] = P[["trend","g","mkt","sep","hlw","lm"]].median(axis=1, skipna=True)
P["lo"] = P[["trend","g","mkt","sep","hlw","lm"]].min(axis=1, skipna=True)
P["hi"] = P[["trend","g","mkt","sep","hlw","lm"]].max(axis=1, skipna=True)
P["n"]  = P[["trend","g","mkt","sep","hlw","lm"]].notna().sum(axis=1)

infl = core_yoy_x.reindex(idx)
P["ff"] = ff.reindex(idx)
P["infl"] = infl
P["real_ff"] = P["ff"] - P["infl"]
P["stance"] = P["real_ff"] - P["composite"]
P["neutral_nom"] = P["composite"] + P["infl"]
P["neutral_nom_lo"] = P["lo"] + P["infl"]
P["neutral_nom_hi"] = P["hi"] + P["infl"]

rec = raw["USREC"].reindex(idx).fillna(0)

# ---------- diagnostics ----------
pd.set_option("display.width", 200)
print(P[["trend","g","mkt","sep","hlw","lm","composite","lo","hi","real_ff","stance"]].tail(4).round(2).to_string())
print("\ndecades (composite / real_ff):")
print(P[["composite","real_ff"]].resample("10YS").mean().round(2).to_string())
last = P.dropna(subset=["composite"]).iloc[-1]
print(f"\nLATEST {P.dropna(subset=['composite']).index[-1].date()}: r*={last['composite']:.2f}  "
      f"range {last['lo']:.2f}..{last['hi']:.2f}  realFF={last['real_ff']:.2f}  "
      f"stance={last['stance']*100:.0f}bp  nom neutral={last['neutral_nom']:.2f}  FF={last['ff']:.2f}  infl={last['infl']:.2f}")

# ---------- payload ----------
def ser(col, r=3):
    return [None if pd.isna(v) else round(float(v), r) for v in P[col]]

payload = {
    "labels": [d.strftime("%Y-%m") for d in idx],
    "rec":    [int(v) for v in rec],
    "series": {k: ser(k) for k in ["trend","g","mkt","sep","hlw","lm","composite","lo","hi",
                                   "ff","infl","real_ff","stance","neutral_nom","neutral_nom_lo","neutral_nom_hi"]},
    "latest": {
        "date": P.dropna(subset=["composite"]).index[-1].strftime("%b %Y"),
        "rstar": round(float(last["composite"]), 2),
        "lo": round(float(last["lo"]), 2), "hi": round(float(last["hi"]), 2),
        "real_ff": round(float(last["real_ff"]), 2),
        "stance_bp": round(float(last["stance"]) * 100),
        "neutral_nom": round(float(last["neutral_nom"]), 2),
        "ff": round(float(last["ff"]), 2), "infl": round(float(last["infl"]), 2),
    },
    "pillars": [],
    "build": dt.date.today().strftime("%d %b %Y"),
}

pillar_meta = [
    ("hlw",  "Holston–Laubach–Williams", "NY Fed · quarterly Kalman state-space (post-COVID spec)", hlw),
    ("lm",   "Lubik–Matthes",            "Richmond Fed · TVP-VAR median", lm),
    ("sep",  "FOMC longer-run dot − 2%", "SEP median longer-run funds rate less inflation target", sep),
    ("mkt",  "Market 5y5y real forward", "TIPS: 2×10y − 5y CMT, monthly avg (incl. term premium)", raw["DFII10"].resample("MS").mean()*2 - raw["DFII5"].resample("MS").mean()),
    ("g",    "Potential growth anchor",  "CBO real potential GDP, y/y (r* ≈ g)", g_yoy),
    ("trend","Realized real-rate trend", "In-house one-sided Kalman local level of FF − core PCE y/y", trend),
]
for key, name, src, s in pillar_meta:
    s = s.dropna()
    payload["pillars"].append({
        "key": key, "name": name, "src": src,
        "asof": s.index[-1].strftime("%b %Y") if key != "hlw" and key != "lm" else s.index[-1].to_period("Q").strftime("%YQ%q"),
        "val": round(float(s.iloc[-1]), 2),
        "nom": round(float(s.iloc[-1]) + 2.0, 2),
    })

pd.to_pickle(P, f"{WORK}/panel.pkl")
json.dump(payload, open(f"{WORK}/payload.json", "w"))
print("\npayload written,", len(payload["labels"]), "months,",
      len(json.dumps(payload)) // 1024, "KB")


# ========================= FETCH OUTCOMES =========================
import requests, time, pandas as pd, numpy as np

ua = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
KEY = FRED_API_KEY
FRED = "https://api.stlouisfed.org/fred/series/observations"

# ---- S&P 500 via Yahoo daily -> month-end last (avoids the silent-omission bug in monthly candles) ----
sp_m = fetch_sp500_monthly(ua)
print(f"S&P 500 monthly: {len(sp_m)} obs {sp_m.index[0].date()} -> {sp_m.index[-1].date()} last={sp_m.iloc[-1]:.0f}")

# ---- FRED outcome series ----
def fred(sid, start="1960-01-01"):
    r = requests.get(FRED, params={"series_id":sid,"api_key":KEY,"file_type":"json","observation_start":start}, timeout=60)
    obs = r.json()["observations"]
    s = pd.Series({o["date"]:(float(o["value"]) if o["value"]!="." else np.nan) for o in obs})
    s.index = pd.to_datetime(s.index); return s.dropna()

series = {}
for sid in ["CPIAUCSL","GDPC1","PAYEMS","INDPRO"]:
    s = fred(sid); series[sid]=s
    print(f"{sid:10s} {len(s):5d} obs {s.index[0].date()} -> {s.index[-1].date()} last={s.iloc[-1]:.2f}")

# real consumption: nominal PCE / PCE price index (long history; PCEC96 only starts 2007 on FRED)
pce = fred("PCE"); pcepi = fred("PCEPI")
series["RPCE"] = (pce/pcepi*100.0).dropna()
print(f"RPCE       {len(series['RPCE']):5d} obs  built from PCE/PCEPI  yoy last={ (series['RPCE'].pct_change(12)*100).iloc[-1]:.2f}%")

series["SP500"] = sp_m
pd.to_pickle(series, f"{WORK}/outcomes.pkl")
print("saved outcomes")


# ========================= FORWARD-OUTCOME ANALYSIS =========================
import pandas as pd, numpy as np, json, datetime as dt

P   = pd.read_pickle(f"{WORK}/panel.pkl")     # monthly, 1965+, has 'stance','real_ff','ff','infl','composite','hlw'
OUT = pd.read_pickle(f"{WORK}/outcomes.pkl")
rng = np.random.default_rng(42)

idx = P.index
stance = P["stance"]

# ================= build aligned monthly outcome levels (log for growth math where sensible) =================
sp   = OUT["SP500"].reindex(idx)                         # month-end S&P
cpi  = OUT["CPIAUCSL"].reindex(idx)
payems = OUT["PAYEMS"].reindex(idx)
indpro = OUT["INDPRO"].reindex(idx)
rpce   = OUT["RPCE"].reindex(idx)
sp_real = (sp / cpi)                                      # CPI-deflated price index

MONTHLY = {
    "sp_nom":  ("S&P 500 (nominal price)",  sp,     "ret"),
    "sp_real": ("S&P 500 (real price)",     sp_real,"ret"),
    "rpce":    ("Real consumption",         rpce,   "grw"),
    "payems":  ("Nonfarm payrolls",         payems, "grw"),
    "indpro":  ("Industrial production",    indpro, "grw"),
}

def fwd_change(level, h, annualize):
    """Forward change from t to t+h. ret/grw both use (t+h)/t - 1; annualized to %/yr if requested."""
    fut = level.shift(-h)
    r = fut / level - 1.0
    if annualize:
        r = (1.0 + r) ** (12.0 / h) - 1.0
    return r * 100.0

# ================= HAC (Newey-West) difference-in-means via dummy regression =================
def hac_diff(y, d, L):
    """y: forward outcome, d: 0/1 restrictive dummy. Returns (diff, t, mean1, mean0, n1, n0)."""
    m = (~y.isna()) & (~d.isna())
    yy = y[m].values.astype(float); dd = d[m].values.astype(float)
    n = len(yy)
    X = np.column_stack([np.ones(n), dd])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ yy)
    e = yy - X @ beta
    # Bartlett-kernel HAC
    S = (X * e[:,None]).T @ (X * e[:,None])
    for l in range(1, L+1):
        w = 1.0 - l/(L+1.0)
        Xe = X * e[:,None]
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    diff = beta[1]; se = np.sqrt(V[1,1])
    return diff, diff/se, yy[dd==1].mean(), yy[dd==0].mean(), int((dd==1).sum()), int((dd==0).sum())

# ================= moving-block bootstrap CI for the difference in means =================
def block_boot_diff(y, d, L_block=24, reps=3000):
    m = (~y.isna()) & (~d.isna())
    yy = y[m].values.astype(float); dd = d[m].values.astype(float)
    n = len(yy); nb = int(np.ceil(n / L_block))
    starts_max = n - L_block
    diffs = np.empty(reps)
    for r in range(reps):
        s = rng.integers(0, starts_max+1, size=nb)
        yb = np.concatenate([yy[i:i+L_block] for i in s])[:n]
        db = np.concatenate([dd[i:i+L_block] for i in s])[:n]
        if db.sum() < 3 or (1-db).sum() < 3:
            diffs[r] = np.nan; continue
        diffs[r] = yb[db==1].mean() - yb[db==0].mean()
    diffs = diffs[~np.isnan(diffs)]
    return np.percentile(diffs, 5), np.percentile(diffs, 95), (diffs>0).mean()

# restrictive dummy on the composite stance
D = (stance > 0).astype(float); D[stance.isna()] = np.nan
D_hlw = ((P["real_ff"] - P["hlw"]) > 0).astype(float)     # robustness: HLW-only stance

# ================= monthly-variable results =================
results = {}
for key,(name, lvl, kind) in MONTHLY.items():
    ann = (kind == "grw")            # annualize macro growth; keep stock returns as horizon totals
    row = {"name":name, "kind":kind, "h":{}}
    for h in (6, 12):
        y = fwd_change(lvl, h, annualize=ann)
        L = int(round(1.3*h)) + 3
        diff, t, m1, m0, n1, n0 = hac_diff(y, D, L)
        lo, hi, pgt = block_boot_diff(y, D)
        # medians
        mm = (~y.isna()) & (~D.isna())
        med1 = np.median(y[mm][D[mm]==1]); med0 = np.median(y[mm][D[mm]==0]); medall = np.median(y[mm])
        # HLW-only robustness diff (means)
        mmh = (~y.isna()) & (~D_hlw.isna())
        dh = D_hlw[mmh]; yh = y[mmh]
        diff_hlw = yh[dh==1].mean() - yh[dh==0].mean()
        # hit rate for stocks
        hit1 = float((y[mm][D[mm]==1] > 0).mean()); hit0 = float((y[mm][D[mm]==0] > 0).mean())
        row["h"][h] = dict(mean_restr=m1, mean_acc=m0, mean_all=float(np.mean(y[mm])),
                           med_restr=med1, med_acc=med0, med_all=medall,
                           diff=diff, t=t, boot_lo=lo, boot_hi=hi, boot_p=pgt,
                           n_restr=n1, n_acc=n0, diff_hlw=diff_hlw,
                           hit_restr=hit1, hit_acc=hit0)
    results[key] = row

# ================= GDP (quarterly) =================
gdp = OUT["GDPC1"]
# stance at quarter = average of the 3 months in that quarter
stq = stance.resample("QS").mean()
gdp_q = gdp.copy(); gdp_q.index = gdp_q.index.to_period("Q").to_timestamp()  # align to quarter-start
def fwd_q(level, q, annualize=True):
    fut = level.shift(-q); r = fut/level - 1.0
    if annualize: r = (1.0+r)**(4.0/q)-1.0
    return r*100.0
gdp_row = {"name":"Real GDP","kind":"grw","h":{}}
Dq = (stq>0).astype(float); Dq[stq.isna()]=np.nan
for h,q in ((6,2),(12,4)):
    y = fwd_q(gdp_q, q).reindex(stq.index)
    L = int(round(1.3*q))+1
    diff,t,m1,m0,n1,n0 = hac_diff(y, Dq, L)
    mm=(~y.isna())&(~Dq.isna())
    gdp_row["h"][h]=dict(mean_restr=m1,mean_acc=m0,mean_all=float(np.mean(y[mm])),
                         med_restr=float(np.median(y[mm][Dq[mm]==1])),med_acc=float(np.median(y[mm][Dq[mm]==0])),
                         med_all=float(np.median(y[mm])),diff=diff,t=t,n_restr=n1,n_acc=n0,
                         boot_lo=np.nan,boot_hi=np.nan,boot_p=np.nan,diff_hlw=np.nan,hit_restr=np.nan,hit_acc=np.nan)
results["gdp"]=gdp_row

# ================= recession share in forward window =================
rec = pd.read_pickle(f"{WORK}/fred_raw.pkl")["USREC"].reindex(idx).fillna(0)
def fwd_rec_share(h):
    # share of months t+1..t+h flagged recession
    fr = pd.concat([rec.shift(-k) for k in range(1,h+1)], axis=1).mean(axis=1)
    mm=(~fr.isna())&(~D.isna())
    return float(fr[mm][D[mm]==1].mean()), float(fr[mm][D[mm]==0].mean())
rec_share = {h:fwd_rec_share(h) for h in (6,12)}

# ================= episode structure =================
sign = (stance>0).astype(int)
runs = []
i=0; s=sign.values
while i < len(s):
    if s[i]==1:
        j=i
        while j<len(s) and s[j]==1: j+=1
        runs.append((idx[i], idx[j-1], j-i))
        i=j
    else: i+=1
n_episodes = len(runs)
months_restr = int(sign.sum())
print(f"Restrictive months: {months_restr}/{len(sign)} ({100*months_restr/len(sign):.0f}%) across {n_episodes} distinct episodes")
print("Episodes (onset -> end, months):")
for a,b,n in runs: print(f"   {a.strftime('%Y-%m')} -> {b.strftime('%Y-%m')}  ({n}m)")

# ================= event-study average paths around ONSET (>=0 -> >0) =================
onsets = [a for (a,b,n) in runs if n>=3]      # require episode length >=3m to be a real regime
def path(level, lo=-6, hi=12, logidx=True):
    mat=[]
    for o in onsets:
        if o not in level.index: continue
        pos = level.index.get_loc(o)
        if pos+lo<0 or pos+hi>=len(level): continue
        seg = level.iloc[pos+lo:pos+hi+1].values.astype(float)
        base = seg[-lo]  # value at t=0
        if base is None or np.isnan(base) or base==0: continue
        mat.append((seg/base - 1.0)*100.0)     # cumulative % change vs onset
    mat=np.array(mat)
    return np.nanmean(mat,axis=0), mat.shape[0]

ev_x = list(range(-6,13))
ev = {}
for key,lvl in [("sp_nom",sp),("indpro",indpro),("payems",payems),("rpce",rpce)]:
    mean_path, k = path(lvl)
    ev[key] = [None if np.isnan(v) else round(float(v),2) for v in mean_path]
ev["_n"] = k

# ---- conditional forward trajectory: from EVERY restrictive vs accommodative month, avg cumulative path 0..12 ----
def cond_path(level, dummy, hmax=12):
    restr=[]; acc=[]
    dv = dummy.values
    for pos in range(len(level)-hmax):
        base = level.iloc[pos]
        if base is None or np.isnan(base) or base==0 or np.isnan(dv[pos]): continue
        seg = (level.iloc[pos:pos+hmax+1].values.astype(float)/base - 1.0)*100.0
        (restr if dv[pos]==1 else acc).append(seg)
    return np.nanmean(np.array(restr),axis=0), np.nanmean(np.array(acc),axis=0)
fwd_path = {}
for key,lvl in [("indpro",indpro),("payems",payems),("rpce",rpce),("sp_nom",sp)]:
    pr,pa = cond_path(lvl, D)
    fwd_path[key] = {"restr":[round(float(v),2) for v in pr], "acc":[round(float(v),2) for v in pa]}

# ================= dose-response: forward 12m outcome vs stance bucket =================
def dose(lvl, kind):
    ann = (kind=="grw")
    y = fwd_change(lvl, 12, annualize=ann)
    df = pd.DataFrame({"stance":stance,"y":y}).dropna()
    df = df[df["stance"]>0]
    # terciles of positive stance
    q = df["stance"].quantile([1/3,2/3]).values
    b1 = df[df["stance"]<=q[0]]["y"].mean()
    b2 = df[(df["stance"]>q[0])&(df["stance"]<=q[1])]["y"].mean()
    b3 = df[df["stance"]>q[1]]["y"].mean()
    return [round(b1,2),round(b2,2),round(b3,2)], [round(q[0],2),round(q[1],2)]
dose_res = {}
for key,(name,lvl,kind) in MONTHLY.items():
    if key in ("sp_nom","indpro","payems"):
        vals,cuts = dose(lvl,kind); dose_res[key]={"name":name,"vals":vals,"cuts":cuts}

# scatter data (stance vs fwd-12m) for S&P nominal and IP
def scatter(lvl, kind):
    ann=(kind=="grw"); y=fwd_change(lvl,12,annualize=ann)
    df=pd.DataFrame({"x":stance,"y":y}).dropna()
    return [[round(float(a),3),round(float(b),2)] for a,b in zip(df["x"],df["y"])]
scat = {"sp_nom":scatter(sp,"ret"), "indpro":scatter(indpro,"grw")}

# ================= current reading =================
cur = P.dropna(subset=["stance"]).iloc[-1]
current = {"date":P.dropna(subset=['stance']).index[-1].strftime("%b %Y"),
           "stance_bp":round(float(cur["stance"])*100), "restrictive": bool(cur["stance"]>0)}

# ================= assemble payload =================
def clean(d):
    out={}
    for k,v in d.items():
        out[k]=None if (isinstance(v,float) and (np.isnan(v))) else (round(v,3) if isinstance(v,(int,float,np.floating)) else v)
    return out

payload = {
    "build": dt.date.today().strftime("%d %b %Y"),
    "sample": f"{idx[0].strftime('%b %Y')}–{idx[-1].strftime('%b %Y')}",
    "n_episodes": n_episodes, "months_restr": months_restr, "months_total": len(sign),
    "episodes": [[a.strftime('%Y-%m'), b.strftime('%Y-%m'), n] for (a,b,n) in runs],
    "results": {k:{"name":v["name"],"kind":v["kind"],
                   "h":{str(h):clean(v["h"][h]) for h in v["h"]}} for k,v in results.items()},
    "rec_share": {str(h):[round(rec_share[h][0],3),round(rec_share[h][1],3)] for h in rec_share},
    "event": {"x":ev_x, "paths":{k:ev[k] for k in ev if k!="_n"}, "n":ev["_n"]},
    "fwd_path": {"x":list(range(0,13)), "vars":fwd_path},
    "dose": dose_res, "scatter": scat, "current": current,
}
json.dump(payload, open(f"{WORK}/analysis_payload.json","w"))

# ================= console summary table =================
print("\n===== FORWARD OUTCOMES: restrictive (stance>0) vs accommodative, mean =====")
order=["sp_nom","sp_real","gdp","rpce","payems","indpro"]
unit={"ret":"% total","grw":"%/yr ann"}
for k in order:
    r=results[k]
    for h in (6,12):
        d=r["h"][h]
        tag = unit[r["kind"]]
        print(f"{r['name']:26s} {h:2d}m  restr={d['mean_restr']:6.2f}  acc={d['mean_acc']:6.2f}  "
              f"diff={d['diff']:6.2f} (t={d['t']:5.2f})  [{tag}]  n_r={d['n_restr']}")
print("\nRecession share of forward window:  6m restr=%.0f%% acc=%.0f%%   12m restr=%.0f%% acc=%.0f%%"%(
    rec_share[6][0]*100,rec_share[6][1]*100,rec_share[12][0]*100,rec_share[12][1]*100))
print("\nDose-response (fwd 12m by tercile of positive stance):")
for k,v in dose_res.items():
    print(f"   {v['name']:24s} shallow={v['vals'][0]:6.2f}  mid={v['vals'][1]:6.2f}  deep={v['vals'][2]:6.2f}  cuts={v['cuts']}")
print(f"\nCurrent stance {current['date']}: {current['stance_bp']}bp  restrictive={current['restrictive']}")
print("payload written", len(json.dumps(payload))//1024,"KB")


# ========================= COMBINED TEMPLATE =========================
COMBINED_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>US Neutral Rate Monitor — r* &amp; Policy Consequences | Acheron Insights</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{--paper:#F4F6F9;--card:#FFFFFF;--ink:#1E2A38;--navy:#1F4E79;--blue:#2F6FB0;--gold:#C0891E;--red:#C0392B;--green:#2E7D52;--muted:#64748B;--line:rgba(30,42,56,0.12);}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--paper);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:1180px;margin:0 auto;padding:34px 26px 60px;}
  [hidden]{display:none!important;}
  .eyebrow{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:12px;letter-spacing:0.22em;color:var(--navy);text-transform:uppercase;}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:32px;letter-spacing:-0.01em;margin:6px 0 4px;}
  .sub{color:var(--muted);font-size:14.5px;max-width:900px;line-height:1.5;}
  .meta{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--muted);margin:10px 0 0;letter-spacing:0.02em;}
  .masthead{padding-bottom:16px;}
  .tabs{display:flex;gap:4px;border-bottom:2px solid var(--ink);margin-bottom:20px;}
  .tabbtn{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13.5px;letter-spacing:0.03em;background:transparent;border:1px solid var(--line);border-bottom:none;color:var(--muted);padding:10px 20px;cursor:pointer;border-radius:3px 3px 0 0;margin-bottom:-2px;transition:all .12s;}
  .tabbtn:hover{color:var(--ink);background:rgba(31,30,27,0.03);}
  .tabbtn.active{color:#FFFFFF;background:var(--ink);border-color:var(--ink);}
  .tabsub{color:var(--muted);font-size:13.5px;line-height:1.5;max-width:900px;margin-bottom:2px;}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:14px 0 28px;}
  .kpis.kpis5{grid-template-columns:repeat(5,1fr);gap:12px;}
  .kpi{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--ink);padding:14px 16px 13px;}
  .kpi.accent-o{border-top-color:var(--gold);}
  .kpi.o{border-top-color:var(--red);}
  .kpi.accent-t{border-top-color:var(--blue);}
  .kpi.t{border-top-color:var(--green);}
  .kpi .lbl{font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);font-weight:600;line-height:1.3;}
  .kpi .val{font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:600;margin-top:6px;line-height:1.1;}
  .kpi .note{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--muted);margin-top:5px;line-height:1.35;}
  .pill{display:inline-block;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:11px;letter-spacing:0.14em;padding:4px 10px;margin-top:8px;border-radius:2px;color:#FFFFFF;}
  #o_kpis .kpi .val{font-size:23px;}
  #o_kpis .kpi .lbl{font-size:10.5px;letter-spacing:0.1em;}
  #o_kpis .kpi .note{font-size:10.5px;}
  .panel{background:var(--card);border:1px solid var(--line);padding:20px 20px 15px;margin-bottom:24px;}
  .panel h2{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;letter-spacing:-0.005em;}
  .panel .desc{font-size:13px;color:var(--muted);margin:4px 0 14px;line-height:1.5;max-width:920px;}
  .chartbox{position:relative;height:380px;}
  .chartbox.short{height:300px;}
  .chartbox.sh{height:280px;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
  .mini{position:relative;height:230px;}
  .mini .cap{font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;margin-bottom:2px;}
  .mini .capn{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-bottom:6px;}
  .lgnd{display:flex;gap:18px;align-items:center;font-size:12px;color:var(--muted);margin:2px 0 12px;}
  .lgnd span{display:inline-flex;align-items:center;gap:6px;}
  .lsw{width:14px;height:3px;border-radius:2px;display:inline-block;}
  table{width:100%;border-collapse:collapse;font-size:13.5px;}
  th{font-family:'Space Grotesk',sans-serif;font-size:11.5px;letter-spacing:0.1em;text-transform:uppercase;text-align:left;color:var(--muted);border-bottom:2px solid var(--ink);padding:8px 10px;}
  th.num,td.num{text-align:right;}
  td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top;}
  td.num{font-family:'IBM Plex Mono',monospace;font-weight:500;}
  td .src{display:block;font-size:12px;color:var(--muted);margin-top:2px;}
  td.grp{font-weight:600;font-family:'Space Grotesk',sans-serif;}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:baseline;}
  tr.comp td{border-top:2px solid var(--ink);border-bottom:none;font-weight:600;}
  tr.comp td.num{font-weight:600;}
  .neg{color:var(--red);}.pos{color:var(--green);}
  .sig{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);}
  .callout{display:flex;gap:20px;flex-wrap:wrap;margin-top:6px;}
  .co{flex:1;min-width:240px;background:rgba(192,57,43,0.06);border-left:3px solid var(--red);padding:12px 14px;}
  .co.t{background:rgba(46,125,82,0.06);border-left-color:var(--green);}
  .co .h{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;margin-bottom:4px;}
  .co .b{font-size:12.5px;color:#334155;line-height:1.55;}
  .exportbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:12px 0 2px;}
  .exlbl{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);}
  .csvbtn{font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:600;color:var(--ink);background:var(--card);border:1px solid var(--line);padding:5px 11px;border-radius:3px;cursor:pointer;transition:all .12s;}
  .csvbtn:hover{background:var(--ink);color:#FFFFFF;border-color:var(--ink);}
  .csvbtn:active{transform:translateY(1px);}
  .method{border-top:2px solid var(--ink);padding-top:18px;margin-top:8px;}
  .method h3{font-family:'Space Grotesk',sans-serif;font-size:14px;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px;}
  .method p{font-size:13px;color:#334155;line-height:1.65;max-width:980px;margin-bottom:10px;}
  .method .foot{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-top:14px;}
  @media(max-width:820px){.kpis,.kpis.kpis5{grid-template-columns:repeat(2,1fr);}.grid2{grid-template-columns:1fr;}h1{font-size:25px;}.chartbox{height:300px;}.tabs{flex-wrap:wrap;}}
</style>
</head>
<body>
<div class="wrap">
<header class="masthead">
  <div class="eyebrow">Acheron Insights · Policy &amp; Rates</div>
  <h1>US Neutral Rate Monitor</h1>
  <div class="sub">A six-model composite estimate of r*, the nominal policy corridor and current stance — and the historical forward consequences of restrictive policy for equities and the real economy.</div>
</header>
<nav class="tabs">
  <button class="tabbtn" data-tab="rates">Neutral Rate</button>
  <button class="tabbtn" data-tab="outcomes">Forward Consequences</button>
</nav>
<section class="tab" data-tab="rates">
  <p class="tabsub">Real r*, monthly since 1965 — six independent estimates, the nominal corridor against effective fed funds, and the resulting policy stance.</p>
  <div class="meta" id="meta"></div>
  <div class="exportbar"><span class="exlbl">⤓ Download</span>
    <button class="csvbtn" data-csv="rates_panel">Monthly r* panel</button>
  </div>


<div class="kpis" id="kpis"></div>

<div class="panel">
  <h2>Six estimates of the natural rate</h2>
  <div class="desc">Real r*, monthly since 1965. Grey band spans the min–max of available estimates; bold line is the cross-sectional median (composite). Shaded columns are NBER recessions.</div>
  <div class="chartbox"><canvas id="c1"></canvas></div>
</div>

<div class="panel">
  <h2>Fed funds vs the neutral corridor</h2>
  <div class="desc">Effective fed funds against the nominal neutral rate, defined as composite r* plus prevailing core PCE inflation. Funds rate above the corridor = restrictive; below = accommodative.</div>
  <div class="chartbox short"><canvas id="c2"></canvas></div>
</div>

<div class="panel">
  <h2>Policy stance — real funds rate minus composite r*</h2>
  <div class="desc">Gap between the realized real funds rate (EFFR less core PCE y/y) and the composite natural rate, in percentage points. Orange = restrictive territory, teal = accommodative.</div>
  <div class="chartbox short"><canvas id="c3"></canvas></div>
</div>

<div class="panel">
  <h2>Where each model sits</h2>
  <div class="desc">Latest reading per pillar. Implied nominal assumes inflation at the 2% target; the corridor chart above instead uses prevailing inflation.</div>
  <table id="ptable">
    <thead><tr><th>Estimate</th><th>As of</th><th class="num">Real r*</th><th class="num">Implied nominal</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="method">
  <h3>Methodology</h3>
  <p id="m1"></p>
  <p id="m2"></p>
  <p id="m3"></p>
  <div class="foot" id="foot"></div>
</div>


</section>
<section class="tab" data-tab="outcomes" hidden>
  <p class="tabsub">What historically follows a positive stance (real fed funds above composite r*): conditional 6- and 12-month paths for markets and the real economy, with HAC inference and depth analysis.</p>
  <div class="meta" id="o_meta"></div>
  <div class="exportbar"><span class="exlbl">⤓ Download</span>
    <button class="csvbtn" data-csv="results">Conditional results</button>
    <button class="csvbtn" data-csv="fwdpaths">Forward paths</button>
    <button class="csvbtn" data-csv="dose">Dose-response</button>
    <button class="csvbtn" data-csv="scatter">Stance vs fwd-12m</button>
  </div>


<div class="kpis kpis5" id="o_kpis"></div>

<div class="panel">
  <h2>Forward trajectory by policy regime</h2>
  <div class="desc">Average cumulative path over the next 12 months, measured from every month classified restrictive (real FF &gt; r*) versus every accommodative month, indexed to zero at the starting month. The gap at month 12 is the conditional drag.</div>
  <div class="lgnd">
    <span><span class="lsw" style="background:#C0392B"></span> Restrictive start (stance &gt; 0)</span>
    <span><span class="lsw" style="background:#2E7D52"></span> Accommodative / neutral start (stance ≤ 0)</span>
  </div>
  <div class="grid2">
    <div><div class="mini"><canvas id="fp_indpro"></canvas></div></div>
    <div><div class="mini"><canvas id="fp_payems"></canvas></div></div>
    <div><div class="mini"><canvas id="fp_rpce"></canvas></div></div>
    <div><div class="mini"><canvas id="fp_sp"></canvas></div></div>
  </div>
</div>

<div class="panel">
  <h2>Conditional forward outcomes — 6 &amp; 12 months</h2>
  <div class="desc">Mean forward growth of real activity, annualized, in the 6 and 12 months after each regime month. Every macro variable is weaker following restrictive policy, and the gap widens from 6m to 12m — the long-lag signature.</div>
  <div class="chartbox sh"><canvas id="bars"></canvas></div>
  <div class="callout">
    <div class="co"><div class="h" id="co1h"></div><div class="b" id="co1b"></div></div>
    <div class="co t"><div class="h" id="co2h"></div><div class="b" id="co2b"></div></div>
  </div>
</div>

<div class="panel">
  <h2>Depth of restriction matters more than its sign</h2>
  <div class="desc">The binary split understates the story: shallow restriction is benign, but deep restriction (real FF well above r*) is where activity actually contracts. Left — forward 12m industrial-production growth split by tercile of positive stance. Right — the continuous relationship across all months.</div>
  <div class="grid2">
    <div><div class="mini" style="height:270px"><canvas id="dose"></canvas></div></div>
    <div><div class="mini" style="height:270px"><canvas id="scatter"></canvas></div></div>
  </div>
  <div class="callout">
    <div class="co"><div class="h">Equities: the tail, not the sign</div><div class="b" id="eqdose"></div></div>
  </div>
</div>

<div class="panel">
  <h2>Full results</h2>
  <div class="desc">Restrictive vs accommodative conditional means. Difference is restrictive − accommodative; t-statistics use Newey–West HAC standard errors (Bartlett kernel, lag ≈ 1.3×horizon) to correct for overlapping forward windows. 90% CI from a 24-month moving-block bootstrap. With only 14 distinct episodes, read the consistency of sign across variables — not any single t — as the signal.</div>
  <table id="rtable">
    <thead><tr><th>Variable</th><th>Horizon</th><th class="num">Restrictive</th><th class="num">Accommod.</th><th class="num">Difference</th><th class="num">HAC t</th><th class="num">Boot 90% CI</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="method">
  <h3>Methodology &amp; caveats</h3>
  <p id="o_m1"></p>
  <p id="o_m2"></p>
  <p id="o_m3"></p>
  <div class="foot" id="o_foot"></div>
</div>


</section>
</div>
<script>
const ROOT = __ROOT_JSON__;
let _rr=false,_ro=false;
function renderRates(){ if(_rr) return; _rr=true; const DATA=ROOT.rates;

const C = {
  ink:'#16324F', orange:'#7FA6CE', teal:'#2F6FB0', mkt:'#9C7A1E', g:'#5E7488',
  lm:'#C79A2B', trend:'#9AA7B4', muted:'#64748B',
  band:'rgba(31,58,95,0.08)', rec:'rgba(30,42,56,0.05)', grid:'rgba(30,42,56,0.09)',
  card:'#FFFFFF'
};
const S = DATA.series, L = DATA.labels, LAST = DATA.latest;

Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
Chart.defaults.color = '#6E6A5E';

// ---------- plugins ----------
const recPlugin = {
  id:'rec',
  beforeDatasetsDraw(chart){
    const {ctx, chartArea:a, scales:{x}} = chart;
    if(!a) return;
    ctx.save(); ctx.fillStyle = C.rec;
    let start = null;
    for(let i=0;i<DATA.rec.length;i++){
      if(DATA.rec[i]===1 && start===null) start = i;
      if((DATA.rec[i]===0 || i===DATA.rec.length-1) && start!==null){
        const x0 = x.getPixelForValue(start), x1 = x.getPixelForValue(i);
        ctx.fillRect(x0, a.top, Math.max(x1-x0,1), a.bottom-a.top);
        start = null;
      }
    }
    ctx.restore();
  }
};
const zeroPlugin = {
  id:'zero',
  afterDatasetsDraw(chart){
    const {ctx, chartArea:a, scales:{y}} = chart;
    if(!a || y.min>0 || y.max<0) return;
    const yp = y.getPixelForValue(0);
    ctx.save(); ctx.strokeStyle = 'rgba(31,30,27,0.5)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(a.left,yp); ctx.lineTo(a.right,yp); ctx.stroke(); ctx.restore();
  }
};

// ---------- shared options ----------
function baseOpts(ymin, ymax, unit){
  return {
    responsive:true, maintainAspectRatio:false, animation:false,
    interaction:{mode:'index', intersect:false},
    plugins:{
      legend:{position:'top', align:'start', labels:{usePointStyle:true, pointStyle:'line', boxWidth:26, padding:14, font:{size:11.5}}},
      tooltip:{
        backgroundColor:'#1F1E1B', titleFont:{family:"'Space Grotesk'", size:12},
        bodyFont:{family:"'IBM Plex Mono'", size:11.5}, padding:10, boxPadding:4,
        filter:(item)=> item.parsed.y!==null && !item.dataset.hideTip,
        callbacks:{ label:(it)=> ` ${it.dataset.label}: ${it.parsed.y>=0?'+':''}${it.parsed.y.toFixed(2)}${unit}` }
      }
    },
    scales:{
      x:{grid:{display:false}, border:{color:'rgba(31,30,27,0.4)'},
         ticks:{maxRotation:0, autoSkip:false, font:{family:"'IBM Plex Mono'", size:10.5},
                callback:function(v){ const lb=L[v]; if(!lb) return null;
                  const [yr,mo]=lb.split('-'); return (mo==='01' && (+yr)%5===0) ? yr : null; }}},
      y:{min:ymin, max:ymax, grid:{color:C.grid}, border:{display:false},
         ticks:{font:{family:"'IBM Plex Mono'", size:10.5}, callback:v=>v+unit}}
    }
  };
}
const line = (label,data,color,w,extra)=>Object.assign(
  {label, data, borderColor:color, borderWidth:w, pointRadius:0, pointHitRadius:6, spanGaps:true, tension:0}, extra||{});

// ---------- chart 1: six estimates ----------
new Chart(document.getElementById('c1'), {
  type:'line',
  data:{ labels:L, datasets:[
    line('Range (min–max)', S.hi, 'rgba(0,0,0,0)', 0, {fill:'+1', backgroundColor:C.band, hideTip:true, order:10}),
    line('_min', S.lo, 'rgba(0,0,0,0)', 0, {hideTip:true, order:10, label:'Range'}),
    line('Composite (median)', S.composite, C.ink, 2.6, {order:0}),
    line('HLW (NY Fed)', S.hlw, C.teal, 1.4, {order:2}),
    line('Lubik–Matthes', S.lm, C.lm, 1.4, {order:3}),
    line('FOMC dot − 2%', S.sep, C.orange, 1.6, {stepped:true, order:4}),
    line('Mkt 5y5y real fwd', S.mkt, C.mkt, 1.4, {order:5}),
    line('Potential growth', S.g, C.g, 1.2, {borderDash:[5,3], order:6}),
    line('Realized-rate trend', S.trend, C.trend, 1.2, {borderDash:[2,2], order:7}),
  ]},
  options:(function(){ const o=baseOpts(undefined, undefined,'%');
    o.plugins.legend.labels.filter = (it)=> it.text!=='_min' && it.text!=='Range';
    return o; })(),
  plugins:[recPlugin, zeroPlugin]
});

// ---------- chart 2: funds vs neutral corridor ----------
new Chart(document.getElementById('c2'), {
  type:'line',
  data:{ labels:L, datasets:[
    line('Corridor', S.neutral_nom_hi, 'rgba(0,0,0,0)', 0, {fill:'+1', backgroundColor:'rgba(47,111,176,0.13)', hideTip:true}),
    line('_nlo', S.neutral_nom_lo, 'rgba(0,0,0,0)', 0, {hideTip:true}),
    line('Nominal neutral (r* + core PCE)', S.neutral_nom, C.teal, 1.8),
    line('Effective fed funds', S.ff, C.ink, 2.2),
  ]},
  options:(function(){ const o=baseOpts(undefined, undefined,'%');
    o.plugins.legend.labels.filter = (it)=> it.text.indexOf('_')!==0 && it.text!=='Corridor';
    return o; })(),
  plugins:[recPlugin]
});

// ---------- chart 3: stance ----------
const pos = S.stance.map(v=> v===null?null:Math.max(v,0));
const neg = S.stance.map(v=> v===null?null:Math.min(v,0));
new Chart(document.getElementById('c3'), {
  type:'line',
  data:{ labels:L, datasets:[
    line('Restrictive', pos, '#C0392B', 1.2, {fill:'origin', backgroundColor:'rgba(192,57,43,0.40)', hideTip:true}),
    line('Accommodative', neg, '#2E7D52', 1.2, {fill:'origin', backgroundColor:'rgba(46,125,82,0.38)', hideTip:true}),
    line('Stance (real FF − r*)', S.stance, 'rgba(0,0,0,0)', 0, {}),
  ]},
  options:(function(){ const o=baseOpts(undefined, undefined,'pp');
    o.plugins.legend.display = false; return o; })(),
  plugins:[recPlugin, zeroPlugin]
});

// ---------- KPIs ----------
const stanceLbl = LAST.stance_bp > 50 ? ['RESTRICTIVE', '#C0392B'] :
                  LAST.stance_bp < -50 ? ['ACCOMMODATIVE', '#2E7D52'] : ['BROADLY NEUTRAL', '#64748B'];
document.getElementById('kpis').innerHTML = `
  <div class="kpi"><div class="lbl">Composite r* (real)</div>
    <div class="val">${LAST.rstar.toFixed(2)}%</div>
    <div class="note">range ${LAST.lo.toFixed(1)} to ${LAST.hi.toFixed(1)} · 6 models</div></div>
  <div class="kpi accent-t"><div class="lbl">Nominal neutral now</div>
    <div class="val">${LAST.neutral_nom.toFixed(2)}%</div>
    <div class="note">r* + core PCE ${LAST.infl.toFixed(1)}% · vs 2%-target: ${(LAST.rstar+2).toFixed(2)}%</div></div>
  <div class="kpi"><div class="lbl">Real fed funds</div>
    <div class="val">${LAST.real_ff>=0?'+':''}${LAST.real_ff.toFixed(2)}%</div>
    <div class="note">EFFR ${LAST.ff.toFixed(2)}% − core PCE y/y</div></div>
  <div class="kpi accent-o"><div class="lbl">Policy stance</div>
    <div class="val">${LAST.stance_bp>0?'+':''}${LAST.stance_bp}bp</div>
    <span class="pill" style="background:${stanceLbl[1]}">${stanceLbl[0]}</span></div>`;

// ---------- pillar table ----------
const swc = {hlw:C.teal, lm:C.lm, sep:C.orange, mkt:C.mkt, g:C.g, trend:C.trend};
let rows = DATA.pillars.map(p=>`
  <tr><td><span class="sw" style="background:${swc[p.key]}"></span><strong>${p.name}</strong><span class="src">${p.src}</span></td>
  <td style="font-family:'IBM Plex Mono',monospace; font-size:12.5px">${p.asof}</td>
  <td class="num">${p.val.toFixed(2)}%</td><td class="num">${p.nom.toFixed(2)}%</td></tr>`).join('');
rows += `<tr class="comp"><td><span class="sw" style="background:${C.ink}"></span>Composite (median of six)</td>
  <td style="font-family:'IBM Plex Mono',monospace; font-size:12.5px">${LAST.date}</td>
  <td class="num">${LAST.rstar.toFixed(2)}%</td><td class="num">${(LAST.rstar+2).toFixed(2)}%</td></tr>`;
document.querySelector('#ptable tbody').innerHTML = rows;

// ---------- meta + methodology ----------
document.getElementById('meta').textContent =
  `DATA THROUGH ${LAST.date.toUpperCase()} · BUILT ${DATA.build.toUpperCase()} · MONTHLY, 1965–PRESENT`;

document.getElementById('m1').innerHTML =
  `<strong>Pillars.</strong> (1) Holston–Laubach–Williams: the NY Fed's joint IS/Phillips-curve Kalman state-space model, post-COVID specification, one-sided estimates, quarterly. (2) Lubik–Matthes: Richmond Fed time-varying-parameter VAR, median estimate. (3) FOMC longer-run median funds-rate projection from the SEP less the 2% inflation target — the Committee's own nominal neutral restated in real terms; stepped between SEP releases. (4) Market-implied: 5y5y forward real yield from TIPS constant maturities (2×10y − 5y, monthly average). This embeds real term and liquidity premia — Kim–Wright 10y term premium is currently ≈0.7pp — so it should be read as an upper-bound pricing of r*. (5) Potential-growth anchor: CBO real potential GDP growth y/y, reflecting the theoretical link r* ≈ g. (6) Realized-rate trend: an in-house one-sided Kalman local-level filter (signal-to-noise 10⁻⁴, ≈8-year effective memory) applied to the ex-post real funds rate, in the spirit of Hamilton–Harris–Hatzius–West long averages.`;
document.getElementById('m2').innerHTML =
  `<strong>Composite &amp; stance.</strong> The composite is the cross-sectional median of whichever pillars exist each month (three by 1965, all six from 2012); the band is the min–max span, a direct read on model disagreement. Real fed funds = effective funds rate less core PCE y/y (final month carried forward one month where PCE lags). Stance = real funds − composite r*; readings beyond ±50bp are labelled restrictive/accommodative. The nominal neutral corridor adds prevailing core inflation to the real corridor — during inflation surges it rises sharply, which is precisely the behind-the-curve read.`;
document.getElementById('m3').innerHTML =
  `<strong>Caveats.</strong> All r* concepts are estimated, not observed; one-sided filters revise as data arrive. The realized-rate trend is an ex-post measure and mechanically dips after inflation surprises (2021–22). Quarterly model pillars are interpolated to monthly and held flat past their last print. Median-of-models is robust to any single pillar's failure but inherits the family's common blind spots (fiscal r*, convenience-yield shifts).`;
document.getElementById('foot').textContent =
  `SOURCES: FRED (FEDFUNDS, PCEPILFE, GDPPOT, DFII5/10, THREEFYTP10, FEDTARMDLR, USREC) · NY FED HLW CURRENT ESTIMATES · RICHMOND FED LUBIK–MATTHES · ACHERON INSIGHTS CALCULATIONS`;

}
function renderOutcomes(){ if(_ro) return; _ro=true; const DATA=ROOT.outcomes;

const C={ink:'#1E2A38',orange:'#C0392B',teal:'#2E7D52',muted:'#64748B',grid:'rgba(30,42,56,0.09)',
         oFill:'rgba(192,57,43,0.14)',tFill:'rgba(46,125,82,0.13)'};
Chart.defaults.font.family="'IBM Plex Sans',sans-serif";
Chart.defaults.color=C.muted;
const R=DATA.results, LAST=DATA.current;

const g=(k,h)=>R[k].h[String(h)];
function fmt(v,d=2,sign=true){return (v>=0&&sign?'+':'')+v.toFixed(d);}

// ---------------- KPI cards ----------------
const ipd=g('indpro',12), rec=DATA.rec_share['12'];
document.getElementById('o_kpis').innerHTML=`
 <div class="kpi"><div class="lbl">Sample</div><div class="val" style="font-size:17px">${DATA.n_episodes} episodes</div>
   <div class="note">${DATA.months_restr}/${DATA.months_total} months restrictive<br>${DATA.sample}</div></div>
 <div class="kpi ${LAST.restrictive?'o':'t'}"><div class="lbl">Current stance</div>
   <div class="val">${LAST.stance_bp>0?'+':''}${LAST.stance_bp}bp</div>
   <div class="note">${LAST.date} · ${LAST.restrictive?'RESTRICTIVE':'ACCOMMODATIVE'}<br>not in the restrictive set</div></div>
 <div class="kpi o"><div class="lbl">Ind. production, 12m</div>
   <div class="val">${fmt(ipd.diff)}pp</div>
   <div class="note">${ipd.mean_restr.toFixed(1)} vs ${ipd.mean_acc.toFixed(1)} %/yr<br>t = ${ipd.t.toFixed(2)} · widest drag</div></div>
 <div class="kpi o"><div class="lbl">Recession density, fwd 12m</div>
   <div class="val">${(rec[0]*100).toFixed(0)}%</div>
   <div class="note">vs ${(rec[1]*100).toFixed(0)}% accommodative<br>≈${(rec[0]/rec[1]).toFixed(1)}× the base rate</div></div>
 <div class="kpi t"><div class="lbl">Equities, 12m nominal</div>
   <div class="val">${fmt(g('sp_nom',12).diff,1)}pp</div>
   <div class="note">${g('sp_nom',12).mean_restr.toFixed(0)}% vs ${g('sp_nom',12).mean_acc.toFixed(0)}%<br>t=${g('sp_nom',12).t.toFixed(2)} · not weaker on avg</div></div>`;

// ---------------- forward-path mini charts ----------------
function miniOpts(unit){return{responsive:true,maintainAspectRatio:false,animation:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{display:false},
    tooltip:{backgroundColor:C.ink,titleFont:{family:"'Space Grotesk'",size:11},bodyFont:{family:"'IBM Plex Mono'",size:11},padding:8,
      callbacks:{title:(it)=>'Month +'+it[0].label,label:(it)=>` ${it.dataset.label}: ${fmt(it.parsed.y)}%`}}},
  scales:{x:{grid:{display:false},border:{color:'rgba(31,30,27,0.35)'},ticks:{font:{family:"'IBM Plex Mono'",size:10},callback:(v)=>DATA.fwd_path.x[v]}},
    y:{grid:{color:C.grid},border:{display:false},ticks:{font:{family:"'IBM Plex Mono'",size:10},callback:(v)=>v+'%'}}}};}
function pathLine(lbl,arr,color,fill){return{label:lbl,data:arr,borderColor:color,borderWidth:2.2,pointRadius:0,pointHitRadius:6,
  fill:fill?'origin':false,backgroundColor:fill,tension:0.15};}
function drawFP(canvas,capId,key,unit,caption){
  const v=DATA.fwd_path.vars[key], x=DATA.fwd_path.x;
  const gap=(v.restr[v.restr.length-1]-v.acc[v.acc.length-1]);
  const host=document.getElementById(canvas).parentElement;
  const cap=document.createElement('div');cap.className='cap';cap.textContent=caption;
  const capn=document.createElement('div');capn.className='capn';
  capn.innerHTML=`12m gap: <b style="color:${gap<0?C.orange:C.teal}">${fmt(gap)}${unit}</b> &nbsp; restr ${fmt(v.restr[12])}${unit} · acc ${fmt(v.acc[12])}${unit}`;
  host.parentElement.insertBefore(cap,host);host.parentElement.insertBefore(capn,host);
  new Chart(document.getElementById(canvas),{type:'line',
    data:{labels:x,datasets:[pathLine('Restrictive',v.restr,C.orange,null),pathLine('Accommodative',v.acc,C.teal,null)]},
    options:miniOpts(unit)});
}
drawFP('fp_indpro',null,'indpro','%','Industrial production');
drawFP('fp_payems',null,'payems','%','Nonfarm payrolls');
drawFP('fp_rpce',null,'rpce','%','Real consumption');
drawFP('fp_sp',null,'sp_nom','%','S&P 500 (nominal price)');

// ---------------- grouped bar: conditional forward outcomes ----------------
const barVars=[['gdp','Real GDP'],['rpce','Consumption'],['payems','Payrolls'],['indpro','Ind. production']];
new Chart(document.getElementById('bars'),{type:'bar',
  data:{labels:barVars.map(v=>v[1]),
    datasets:[
      {label:'Restrictive · 6m',data:barVars.map(v=>g(v[0],6).mean_restr),backgroundColor:'rgba(192,57,43,0.5)',borderColor:C.orange,borderWidth:1},
      {label:'Accommodative · 6m',data:barVars.map(v=>g(v[0],6).mean_acc),backgroundColor:'rgba(46,125,82,0.5)',borderColor:C.teal,borderWidth:1},
      {label:'Restrictive · 12m',data:barVars.map(v=>g(v[0],12).mean_restr),backgroundColor:C.orange},
      {label:'Accommodative · 12m',data:barVars.map(v=>g(v[0],12).mean_acc),backgroundColor:C.teal},
    ]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
    plugins:{legend:{position:'top',align:'start',labels:{usePointStyle:true,pointStyle:'rect',boxWidth:11,padding:12,font:{size:11.5}}},
      tooltip:{backgroundColor:C.ink,bodyFont:{family:"'IBM Plex Mono'",size:11},callbacks:{label:(it)=>` ${it.dataset.label}: ${it.parsed.y.toFixed(2)} %/yr`}}},
    scales:{x:{grid:{display:false},border:{color:'rgba(31,30,27,0.35)'},ticks:{font:{family:"'Space Grotesk'",size:12}}},
      y:{grid:{color:C.grid},border:{display:false},title:{display:true,text:'forward growth, %/yr annualized',font:{size:11}},ticks:{font:{family:"'IBM Plex Mono'",size:10},callback:(v)=>v+'%'}}}}});

document.getElementById('co1h').textContent='Real economy — uniformly weaker';
document.getElementById('co1b').innerHTML=`Following restrictive policy, forward growth is lower for every measure and the shortfall <b>widens with horizon</b>: industrial production ${g('indpro',6).diff.toFixed(2)}pp at 6m → ${g('indpro',12).diff.toFixed(2)}pp at 12m; payrolls ${g('payems',6).diff.toFixed(2)} → ${g('payems',12).diff.toFixed(2)}pp. Consistent with monetary policy's long and variable lags.`;
document.getElementById('co2h').textContent='Equities — no reliable drag';
document.getElementById('co2b').innerHTML=`Average 12m nominal S&P returns are actually <b>${fmt(g('sp_nom',12).diff,1)}pp higher</b> under restrictive policy (t≈${g('sp_nom',12).t.toFixed(1)}), and the positive-return hit rate is near-identical (${(g('sp_nom',12).hit_restr*100).toFixed(0)}% vs ${(g('sp_nom',12).hit_acc*100).toFixed(0)}%). The 1994–2001 restrictive-but-booming stretch dominates the sample.`;

// ---------------- dose-response (IP terciles) ----------------
const dip=DATA.dose.indpro, dpay=DATA.dose.payems;
new Chart(document.getElementById('dose'),{type:'bar',
  data:{labels:['Shallow\n(≤'+dip.cuts[0]+'pp)','Moderate\n('+dip.cuts[0]+'–'+dip.cuts[1]+'pp)','Deep\n(>'+dip.cuts[1]+'pp)'],
    datasets:[
      {label:'Industrial production',data:dip.vals,backgroundColor:[ '#E8A79E','#D46F62',C.orange]},
      {label:'Payrolls',data:dpay.vals,backgroundColor:['#9DC7B0','#5FA07E',C.teal]},
    ]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
    plugins:{legend:{position:'top',align:'start',labels:{usePointStyle:true,pointStyle:'rect',boxWidth:11,padding:10,font:{size:11}}},
      title:{display:true,text:'Forward 12m growth by depth of restriction',font:{family:"'Space Grotesk'",size:12.5,weight:'700'},color:C.ink,padding:{bottom:8}},
      tooltip:{backgroundColor:C.ink,bodyFont:{family:"'IBM Plex Mono'",size:11},callbacks:{label:(it)=>` ${it.dataset.label}: ${it.parsed.y.toFixed(2)} %/yr`}}},
    scales:{x:{grid:{display:false},border:{color:'rgba(31,30,27,0.35)'},ticks:{font:{family:"'IBM Plex Mono'",size:10}}},
      y:{grid:{color:C.grid},border:{display:false},ticks:{font:{family:"'IBM Plex Mono'",size:10},callback:(v)=>v+'%'}}}}});

// ---------------- scatter: fwd-12m IP vs stance, with OLS line ----------------
const pts=DATA.scatter.indpro;
const inR=pts.filter(p=>p[0]>0), inA=pts.filter(p=>p[0]<=0);
// OLS
let n=pts.length,sx=0,sy=0,sxx=0,sxy=0;pts.forEach(p=>{sx+=p[0];sy+=p[1];sxx+=p[0]*p[0];sxy+=p[0]*p[1];});
const b=(n*sxy-sx*sy)/(n*sxx-sx*sx), a=(sy-b*sx)/n;
const xmin=Math.min(...pts.map(p=>p[0])), xmax=Math.max(...pts.map(p=>p[0]));
new Chart(document.getElementById('scatter'),{type:'scatter',
  data:{datasets:[
    {label:'Accommodative month',data:inA.map(p=>({x:p[0],y:p[1]})),backgroundColor:'rgba(46,125,82,0.35)',pointRadius:2.2},
    {label:'Restrictive month',data:inR.map(p=>({x:p[0],y:p[1]})),backgroundColor:'rgba(192,57,43,0.42)',pointRadius:2.2},
    {label:'OLS fit',type:'line',data:[{x:xmin,y:a+b*xmin},{x:xmax,y:a+b*xmax}],borderColor:C.ink,borderWidth:2,pointRadius:0,borderDash:[6,3]},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
    plugins:{legend:{position:'top',align:'start',labels:{usePointStyle:true,boxWidth:8,padding:10,font:{size:10.5},filter:(i)=>i.text!=='OLS fit'}},
      title:{display:true,text:'12m IP growth vs policy stance (slope '+b.toFixed(2)+' %/yr per pp)',font:{family:"'Space Grotesk'",size:12,weight:'700'},color:C.ink,padding:{bottom:8}},
      tooltip:{enabled:false}},
    scales:{x:{grid:{color:C.grid},border:{color:'rgba(31,30,27,0.35)'},title:{display:true,text:'policy stance: real FF − r* (pp)',font:{size:10.5}},ticks:{font:{family:"'IBM Plex Mono'",size:10},callback:(v)=>v+'pp'}},
      y:{grid:{color:C.grid},border:{display:false},title:{display:true,text:'fwd 12m IP, %/yr',font:{size:10.5}},ticks:{font:{family:"'IBM Plex Mono'",size:10},callback:(v)=>v+'%'}}}}});

document.getElementById('eqdose').innerHTML=`Split by depth, 12-month nominal S&P returns run <b>+${DATA.dose.sp_nom.vals[0].toFixed(0)}% / +${DATA.dose.sp_nom.vals[1].toFixed(0)}%</b> for shallow-to-moderate restriction but collapse to <b>+${DATA.dose.sp_nom.vals[2].toFixed(0)}%</b> once real funds sit more than ${DATA.dose.sp_nom.cuts[1]}pp above r*. The equity risk is concentrated in the deep-restriction tail — precisely the Volcker-1980 and 2006–07 configurations — not in restrictive policy as such.`;

// ---------------- results table ----------------
const rows=[['sp_nom','S&P 500 nominal','% tot'],['sp_real','S&P 500 real','% tot'],['gdp','Real GDP','%/yr'],
            ['rpce','Real consumption','%/yr'],['payems','Nonfarm payrolls','%/yr'],['indpro','Industrial production','%/yr']];
let html='';
rows.forEach(([k,nm,u])=>{
  [6,12].forEach((h,i)=>{
    const d=g(k,h);
    const ci=(d.boot_lo!=null)?`[${d.boot_lo.toFixed(1)}, ${d.boot_hi.toFixed(1)}]`:'—';
    html+=`<tr>${i===0?`<td class="grp" rowspan="2">${nm}<br><span class="sig">${u}</span></td>`:''}
      <td class="sig">${h}m</td>
      <td class="num">${d.mean_restr.toFixed(2)}</td>
      <td class="num">${d.mean_acc.toFixed(2)}</td>
      <td class="num ${d.diff<0?'neg':'pos'}">${fmt(d.diff)}</td>
      <td class="num">${d.t.toFixed(2)}</td>
      <td class="num sig">${ci}</td></tr>`;
  });
});
document.querySelector('#rtable tbody').innerHTML=html;

// ---------------- meta + methodology ----------------
document.getElementById('o_meta').textContent=`SAMPLE ${DATA.sample.toUpperCase()} · ${DATA.n_episodes} RESTRICTIVE EPISODES · BUILT ${DATA.build.toUpperCase()}`;
document.getElementById('o_m1').innerHTML=`<strong>Design.</strong> Policy stance = realized real fed funds (EFFR − core PCE y/y) minus the six-model composite r* from the companion monitor. A month is <em>restrictive</em> when stance &gt; 0. For each variable we take forward changes over the next 6 and 12 months — total price return for equities, annualized growth for macro — and compare the mean conditional on the current month's regime. Real GDP is handled on the quarterly grid (2Q / 4Q); consumption, payrolls and industrial production are monthly. The stance is restrictive in ${DATA.months_restr} of ${DATA.months_total} months (${(100*DATA.months_restr/DATA.months_total).toFixed(0)}%) spread across ${DATA.n_episodes} distinct episodes — the 1994–2001 stretch alone contributes 76 of them.`;
document.getElementById('o_m2').innerHTML=`<strong>What holds up.</strong> Every real-activity measure grows more slowly after restrictive months, and the shortfall widens from 6m to 12m — the textbook long-lag pattern. Forward recession density is ${(DATA.rec_share['12'][0]*100).toFixed(0)}% versus ${(DATA.rec_share['12'][1]*100).toFixed(0)}% at 12 months, roughly ${(DATA.rec_share['12'][0]/DATA.rec_share['12'][1]).toFixed(1)}×. Industrial production shows the largest and most reliable drag (bootstrap 90% CI barely clears zero); the sign pattern survives re-defining stance off the HLW model alone. Equities are the exception: average and median forward returns are <em>not</em> lower under restriction — the damage appears only in the deepest tercile of restriction, where 12m IP turns negative and equity returns fall to low single digits.`;
document.getElementById('o_m3').innerHTML=`<strong>Caveats.</strong> This is a conditional characterization, not a causal or tradeable one. (i) <em>Endogeneity</em>: the Fed turns restrictive <em>because</em> growth and inflation are hot, so near-term forward data partly reflect that momentum; the drag is what remains after the lag. (ii) <em>Overlap</em>: forward windows are serially correlated; HAC and block-bootstrap widen the error bars, and with only ${DATA.n_episodes} episodes no single t is decisive — the cross-variable consistency and recession-density gap carry the argument. (iii) <em>Not real-time</em>: the composite is built on revised data and one-sided but not vintage estimates, so this is a historical read, not a backtest of a live signal. (iv) <em>Composite drift</em>: the market and SEP pillars enter only in 2003 and 2012, shifting the zero-line's meaning across eras. <strong>Current relevance:</strong> stance is ${LAST.stance_bp}bp — accommodative — so the economy currently sits in the stronger-growth, low-recession-density basket, not the restrictive one this note characterizes.`;
document.getElementById('o_foot').textContent=`SOURCES: FRED (FEDFUNDS, PCEPILFE, CPIAUCSL, GDPC1, PCE/PCEPI, PAYEMS, INDPRO, USREC) · S&P 500 VIA YAHOO FINANCE (DAILY→MONTHLY) · COMPOSITE r* PER ACHERON NEUTRAL-RATE MONITOR · ACHERON INSIGHTS CALCULATIONS`;

}
function showTab(name){
  document.querySelectorAll('.tab').forEach(function(s){ s.hidden = (s.getAttribute('data-tab')!==name); });
  document.querySelectorAll('.tabbtn').forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-tab')===name); });
  if(name==='rates'){ renderRates(); } else { renderOutcomes(); }
}
document.querySelectorAll('.tabbtn').forEach(function(b){ b.addEventListener('click', function(){ showTab(b.getAttribute('data-tab')); }); });

// ---------------- CSV export ----------------
function toCSV(rows){
  return rows.map(function(r){ return r.map(function(c){
    if(c===null||c===undefined) return '';
    var s=String(c);
    if(/[",\n]/.test(s)) s='"'+s.replace(/"/g,'""')+'"';
    return s;
  }).join(','); }).join('\n');
}
function downloadCSV(fname, rows){
  var blob=new Blob([toCSV(rows)], {type:'text/csv;charset=utf-8;'});
  var url=URL.createObjectURL(blob), a=document.createElement('a');
  a.href=url; a.download=fname; document.body.appendChild(a); a.click();
  document.body.removeChild(a); setTimeout(function(){ URL.revokeObjectURL(url); }, 1000);
}
function csvRatesPanel(){
  var R=ROOT.rates, s=R.series, L=R.labels;
  var cols=[['composite','composite_rstar'],['lo','range_lo'],['hi','range_hi'],['hlw','hlw'],
    ['lm','lubik_matthes'],['sep','fomc_dot_less2pct'],['mkt','mkt_5y5y_real'],['g','pot_growth'],
    ['trend','realized_real_trend'],['ff','eff_ffr'],['infl','core_pce_yoy'],['real_ff','real_ffr'],
    ['stance','stance_realffr_minus_rstar'],['neutral_nom','nominal_neutral'],
    ['neutral_nom_lo','nominal_neutral_lo'],['neutral_nom_hi','nominal_neutral_hi']];
  var rows=[['date'].concat(cols.map(function(c){return c[1];})).concat(['nber_recession'])];
  for(var i=0;i<L.length;i++){ var row=[L[i]];
    cols.forEach(function(c){ row.push(s[c[0]][i]); }); row.push(R.rec[i]); rows.push(row); }
  return rows;
}
function csvResults(){
  var R=ROOT.outcomes.results;
  var rows=[['variable','name','unit','horizon_m','mean_restrictive','mean_accommodative','mean_all',
    'median_restrictive','median_accommodative','diff','hac_t','boot_lo','boot_hi','boot_p_gt0',
    'n_restrictive','n_accommodative','diff_hlw_only','hit_restrictive','hit_accommodative']];
  ['sp_nom','sp_real','gdp','rpce','payems','indpro'].forEach(function(k){ var v=R[k];
    [6,12].forEach(function(h){ var d=v.h[String(h)];
      rows.push([k,v.name,v.kind,h,d.mean_restr,d.mean_acc,d.mean_all,d.med_restr,d.med_acc,d.diff,d.t,
        d.boot_lo,d.boot_hi,d.boot_p,d.n_restr,d.n_acc,d.diff_hlw,d.hit_restr,d.hit_acc]); }); });
  return rows;
}
function csvFwdPaths(){
  var F=ROOT.outcomes.fwd_path, vk=['indpro','payems','rpce','sp_nom'], hdr=['month'];
  vk.forEach(function(k){ hdr.push(k+'_restrictive'); hdr.push(k+'_accommodative'); });
  var rows=[hdr];
  for(var i=0;i<F.x.length;i++){ var row=[F.x[i]];
    vk.forEach(function(k){ row.push(F.vars[k].restr[i]); row.push(F.vars[k].acc[i]); }); rows.push(row); }
  return rows;
}
function csvDose(){
  var D=ROOT.outcomes.dose, rows=[['variable','name','bucket','stance_range_pp','fwd_12m']];
  Object.keys(D).forEach(function(k){ var d=D[k], c=d.cuts;
    rows.push([k,d.name,'shallow','<= '+c[0], d.vals[0]]);
    rows.push([k,d.name,'moderate','('+c[0]+', '+c[1]+']', d.vals[1]]);
    rows.push([k,d.name,'deep','> '+c[1], d.vals[2]]); });
  return rows;
}
function csvScatter(){
  var S=ROOT.outcomes.scatter, rows=[['variable','stance_pp','fwd_12m']];
  Object.keys(S).forEach(function(k){ S[k].forEach(function(p){ rows.push([k,p[0],p[1]]); }); });
  return rows;
}
var CSVMAP={rates_panel:['rstar_monthly_panel.csv',csvRatesPanel],
  results:['forward_outcomes_results.csv',csvResults], fwdpaths:['forward_paths.csv',csvFwdPaths],
  dose:['dose_response.csv',csvDose], scatter:['stance_vs_fwd12m.csv',csvScatter]};
document.querySelectorAll('[data-csv]').forEach(function(btn){
  btn.addEventListener('click', function(){ var m=CSVMAP[btn.getAttribute('data-csv')]; downloadCSV(m[0], m[1]()); });
});

showTab('rates');
</script>
</body>
</html>"""


# ================= ASSEMBLE combined dashboard =================
_root = {"rates":    json.load(open(f"{WORK}/payload.json")),
         "outcomes": json.load(open(f"{WORK}/analysis_payload.json"))}
_final = COMBINED_TEMPLATE.replace("__ROOT_JSON__", json.dumps(_root))
_out = OUT_HTML
open(_out, "w").write(_final)
print("WROTE", _out, len(_final)//1024, "KB")

# ================= publish extras =================
# keep a descriptively-named copy alongside index.html
import shutil
shutil.copyfile(OUT_HTML, os.path.join(PUBLIC, "us_neutral_rate_monitor.html"))

# machine-readable build stamp for monitoring / cache-busting
_stamp = {"built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
          "data_through": _root["rates"]["latest"]["date"],
          "composite_rstar": _root["rates"]["latest"]["rstar"],
          "stance_bp": _root["rates"]["latest"]["stance_bp"]}
json.dump(_stamp, open(os.path.join(PUBLIC, "build.json"), "w"), indent=2)
print("BUILD STAMP", _stamp)
