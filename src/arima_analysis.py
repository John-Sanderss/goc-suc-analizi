# arima_analysis.py
# ARIMA ile toplam suç tahmini: tek ülke, iki ülke, Avrupa geneli
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

SUCLAR = [
    "Assault","Corruption","Cybercrime","Drug offences","Fraud",
    "Homicide","Kidnapping","Money laundering","Organized crime",
    "Rape","Robbery","Sexual violence","Theft"
]

DATA_PATH = "data/merged_goc_suc.csv"
OUTDIR = "results/arima"

def _load_df():
    df = pd.read_csv(DATA_PATH)
    df["yıl"] = pd.to_numeric(df["yıl"], errors="coerce").astype("Int64")
    for col in SUCLAR + ["göç"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["yıl"])

def _ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def arima_single_country(country: str, steps: int = 3, order=(1,1,1)):
    _ensure_outdir()
    df = _load_df()
    dfc = df[df["ülke"] == country].sort_values("yıl").copy()
    dfc["total_crime"] = dfc[SUCLAR].sum(axis=1, min_count=1)
    ts = dfc["total_crime"].dropna()

    fit = ARIMA(ts, order=order).fit()
    fc = fit.forecast(steps=steps)
    last = int(dfc["yıl"].iloc[-1])
    years = list(dfc["yıl"]) + [last + i for i in range(1, steps+1)]
    vals = list(ts.values) + list(fc.values)

    plt.figure(figsize=(10,6))
    plt.plot(dfc["yıl"], ts, label=f"{country} – Gerçek", linewidth=2)
    plt.plot(years, vals, marker="o", linestyle="--", label=f"{country} – Tahmin (+{steps}y)")
    plt.axvline(x=last, color="red", linestyle="--", label="Tahmin Başlangıcı")
    plt.title(f"{country} – Toplam Suç ARIMA {order}")
    plt.xlabel("Yıl"); plt.ylabel("Toplam Suç"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/arima_{country.replace(' ', '_')}.png", dpi=300)
    plt.close()

def arima_two_countries(countries=("Spain","Hungary"), steps: int = 3, order=(1,1,1)):
    _ensure_outdir()
    df = _load_df()
    plt.figure(figsize=(12,6))
    for country in countries:
        dfc = df[df["ülke"] == country].sort_values("yıl").copy()
        if dfc.empty: 
            continue
        dfc["total_crime"] = dfc[SUCLAR].sum(axis=1, min_count=1)
        ts = dfc["total_crime"].dropna()
        fit = ARIMA(ts, order=order).fit()
        fc = fit.forecast(steps=steps)
        last = int(dfc["yıl"].iloc[-1])
        years = list(dfc["yıl"]) + [last + i for i in range(1, steps+1)]
        vals = list(ts.values) + list(fc.values)
        plt.plot(dfc["yıl"], ts, label=f"{country} – Gerçek", linewidth=2)
        plt.plot(years, vals, marker="o", linestyle="--", label=f"{country} – Tahmin")
    plt.title("İki Ülke – Toplam Suç ARIMA")
    plt.xlabel("Yıl"); plt.ylabel("Toplam Suç"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/two_countries.png", dpi=300)
    plt.close()

def arima_europe(steps: int = 3, order=(1,1,1)):
    _ensure_outdir()
    df = _load_df()
    df["total_crime"] = df[SUCLAR].sum(axis=1, min_count=1)
    eu = df.groupby("yıl", as_index=False)["total_crime"].sum().sort_values("yıl")
    ts = eu["total_crime"]
    fit = ARIMA(ts, order=order).fit()
    fc = fit.forecast(steps=steps)
    last = int(eu["yıl"].iloc[-1])
    years = list(eu["yıl"]) + [last + i for i in range(1, steps+1)]
    vals = list(ts.values) + list(fc.values)

    plt.figure(figsize=(10,6))
    plt.plot(eu["yıl"], ts, label="Avrupa – Gerçek", linewidth=2)
    plt.plot(years, vals, marker="o", linestyle="--", label=f"ARIMA Tahmin (+{steps}y)")
    plt.axvline(x=last, color="red", linestyle="--", label="Tahmin Başlangıcı")
    plt.title(f"Avrupa – Toplam Suç ARIMA {order}")
    plt.xlabel("Yıl"); plt.ylabel("Toplam Suç"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/europe.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    arima_single_country("Germany", steps=3)
    arima_two_countries(("Spain","Hungary"), steps=3)
    arima_europe(steps=3)
    print("ARIMA çıktıları: results/arima/")
