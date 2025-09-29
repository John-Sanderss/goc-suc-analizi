# random_forest_analysis.py
# Göç verisi ile suç oranı ilişkisi – birleşik ve sadeleştirilmiş
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

SUCLAR = [
    "Assault","Corruption","Cybercrime","Drug offences","Fraud",
    "Homicide","Kidnapping","Money laundering","Organized crime",
    "Rape","Robbery","Sexual violence","Theft"
]

DATA_PATH = "data/merged_goc_suc.csv"
OUTDIR = "results/rf"

def _load_df():
    df = pd.read_csv(DATA_PATH)
    df["yıl"] = pd.to_numeric(df["yıl"], errors="coerce").astype("Int64")
    for col in SUCLAR + ["göç"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["yıl"])

def _ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def rf_selected_countries(countries=("Spain","Hungary")):
    _ensure_outdir()
    df = _load_df()
    plt.figure(figsize=(12,6))
    for ulke in countries:
        d = df[df["ülke"] == ulke].sort_values("yıl").copy()
        if d.empty: 
            continue
        d["total_crime"] = d[SUCLAR].sum(axis=1, min_count=1)
        X, y = d[["göç"]], d["total_crime"]
        m = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
        yhat = m.predict(X); r2 = r2_score(y, yhat)
        plt.plot(d["yıl"], y, marker="o", label=f"{ulke} – Gerçek")
        plt.plot(d["yıl"], yhat, linestyle="--", label=f"{ulke} – RF (R²={r2:.2f})")
    plt.title("Seçilen Ülkeler – Göçten Toplam Suç Tahmini")
    plt.xlabel("Yıl"); plt.ylabel("Toplam Suç"); plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/selected_countries.png", dpi=300); plt.close()

def rf_all_countries():
    _ensure_outdir()
    df = _load_df()
    rows = []
    for ulke in df["ülke"].unique():
        d = df[df["ülke"] == ulke].sort_values("yıl").copy()
        if d.empty: 
            continue
        d["total_crime"] = d[SUCLAR].sum(axis=1, min_count=1)
        X, y = d[["göç"]], d["total_crime"]
        if y.nunique() < 2: 
            continue
        m = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
        r2 = r2_score(y, m.predict(X))
        rows.append((ulke, r2))
    skor = pd.DataFrame(rows, columns=["Ülke","R2 Skoru"]).sort_values("R2 Skoru", ascending=False)
    plt.figure(figsize=(10,8))
    sns.barplot(data=skor, x="R2 Skoru", y="Ülke", palette="mako")
    plt.title("Ülkelere Göre Göçten Toplam Suç Tahmin Başarısı (RF)")
    plt.xlim(0,1); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/all_countries.png", dpi=300); plt.close()
    skor.to_csv(f"{OUTDIR}/all_countries_r2.csv", index=False)

def rf_europe_by_crimetype():
    _ensure_outdir()
    df = _load_df()
    X = df[["göç"]]
    rows = []
    for suc in SUCLAR:
        y = df[suc]
        if y.nunique() < 2: 
            continue
        m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        rows.append((suc, r2_score(y, m.predict(X))))
    tab = pd.DataFrame(rows, columns=["Suç Türü","R2 Skoru"]).sort_values("R2 Skoru", ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=tab, x="R2 Skoru", y="Suç Türü", palette="viridis")
    plt.title("Avrupa – Göç ile Suç Türleri Tahmin Başarısı (RF)")
    plt.xlim(0,1); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/europe_by_crimetype.png", dpi=300); plt.close()
    tab.to_csv(f"{OUTDIR}/europe_by_crimetype.csv", index=False)

def rf_country_by_crimetype():
    _ensure_outdir()
    df = _load_df()
    rows = []
    for ulke in df["ülke"].unique():
        d = df[df["ülke"] == ulke].copy()
        X = d[["göç"]]
        for suc in SUCLAR:
            y = d[suc]
            if y.nunique() < 2: 
                continue
            m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
            rows.append((ulke, suc, r2_score(y, m.predict(X))))
    tab = pd.DataFrame(rows, columns=["Ülke","Suç Türü","R2 Skoru"])
    tab.to_csv(f"{OUTDIR}/country_by_crimetype.csv", index=False)

    # her ülke için grafik
    for ulke in tab["Ülke"].unique():
        t = tab[tab["Ülke"] == ulke].sort_values("R2 Skoru", ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(data=t, x="R2 Skoru", y="Suç Türü", palette="viridis")
        plt.title(f"{ulke} – Göç ile Suç Türü Tahmin Başarısı (RF)")
        plt.xlim(0,1); plt.tight_layout()
        plt.savefig(f"{OUTDIR}/r2_{ulke.replace(' ','_')}.png", dpi=300); plt.close()

if __name__ == "__main__":
    rf_selected_countries(("Spain","Hungary"))
    rf_all_countries()
    rf_europe_by_crimetype()
    rf_country_by_crimetype()
    print("RF çıktıları: results/rf/")
