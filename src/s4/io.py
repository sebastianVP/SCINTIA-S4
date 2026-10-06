"""Lectura de datasets: S4 reconstruido y variables precursoras."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _a_indice_temporal(df: pd.DataFrame, candidatos=("Tiempo", "datetime", "Datetime", "timestamp", "Unnamed: 0")) -> pd.DataFrame:
    """Usa como DatetimeIndex la primera columna de tiempo disponible."""
    out = df.copy()
    col = next((c for c in candidatos if c in out.columns), None)
    if col is None:
        primera = out.columns[0]
        prueba = pd.to_datetime(out[primera].astype(str).head(1000), errors="coerce")
        if prueba.notna().all():
            col = primera
    if col is None:
        raise ValueError(f"No se encontró columna de tiempo. Columnas del CSV: {list(out.columns)[:10]}")
    out[col] = pd.to_datetime(out[col].astype(str).str.strip(), errors="coerce")
    malos = int(out[col].isna().sum())
    if malos:
        raise ValueError(f"{malos} timestamps inválidos en la columna '{col}'.")
    out = out.set_index(col)
    out.index.name = "Tiempo"
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"{int(out.index.duplicated().sum())} timestamps duplicados.")
    return out

def cargar_s4_reconstruido(ruta, flags=("Fix0", "Fix1", "Fix3", "Fix4", "Fix_Pico")) -> pd.DataFrame:
    """Devuelve DataFrame con S4 y la columna booleana `reconstruido`
    (True si el minuto fue modificado/rellenado por Fix0–Fix4/Fix_Pico)."""
    df = _a_indice_temporal(pd.read_csv(ruta, index_col=None))
    if "S4" not in df.columns:
        raise ValueError("El CSV no tiene columna S4.")
    presentes = [f for f in flags if f in df.columns]
    rec = np.zeros(len(df), dtype=bool)
    for f in presentes:
        rec |= df[f].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    out = pd.DataFrame({"S4": df["S4"].astype(float), "reconstruido": rec}, index=df.index)
    if out["S4"].isna().any():
        raise ValueError("S4 contiene NaN tras la reconstrucción.")
    print(f"S4 reconstruido: {len(out):,} obs | {out.index.min()} -> {out.index.max()} | "
          f"minutos reconstruidos: {rec.sum():,} ({100*rec.mean():.2f} %) | flags: {presentes}")
    return out


def cargar_multivariable(ruta, columnas=("TEC", "ROTEC", "ROTI", "Kp_Index", "Dst_Index",
                                         "AE_Index", "f10.7_Index")) -> pd.DataFrame:
    """Lee el dataset multivariable (DF_FINAL_JICAMARCA_*.csv). NO se usa su S4:
    el S4 de todos los experimentos es el reconstruido, para que la ablación
    compare modelos sobre exactamente el mismo objetivo."""
    df = _a_indice_temporal(pd.read_csv(ruta))
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el multivariable: {faltan}")
    return df[list(columnas)].astype(float)


def recortar_periodo(df: pd.DataFrame, inicio, fin) -> pd.DataFrame:
    fin_dia = pd.Timestamp(fin) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return df.loc[pd.Timestamp(inicio):fin_dia]
