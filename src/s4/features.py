"""Ingeniería de características: codificación cíclica y precursores CAUSALES.

ADVERTENCIA METODOLÓGICA (nueva, importante para la tesis)
----------------------------------------------------------
En DF_FINAL_JICAMARCA_31072026.csv los índices horarios (Dst, AE) aparecen
INTERPOLADOS LINEALMENTE a 1 minuto (p. ej. Dst = -1.4167, -1.4000, -1.3833 …).
Interpolar entre la hora h y la hora h+1 hace que el minuto h:35 "conozca" el
valor de la hora siguiente: es una fuga de información del futuro de hasta
~60 min, del mismo tipo que la observada por el especialista.

Además, el valor de Dst/AE de la hora h es un promedio sobre [h, h+1): solo
existe a partir de h+1. Kp (3 h) y F10.7 (diario) tienen el mismo problema.

`causalizar_indice()` reconstruye cada índice para que en el minuto t solo
se use el último valor que ya estaba disponible en t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CICLICAS = ["daily_sin", "daily_cos", "day_of_year_sin", "day_of_year_cos"]
LOCALES = ["TEC", "ROTEC", "ROTI"]
GLOBALES = ["Kp_Index", "Dst_Index", "AE_Index", "f10.7_Index"]

# Conjuntos de entrada de la ablación (el orden replica el pedido del especialista:
# 1) modelo base, 2) + codificación cíclica, 3) + WFL [cambia la pérdida, no las entradas],
# y se extiende con precursores para responder la pregunta sobre ROTI).
CONJUNTOS = {
    "A_S4":            ["S4"],
    "B_S4_ciclica":    ["S4"] + CICLICAS,
    "D_locales":       ["S4"] + CICLICAS + LOCALES,
    "E_completo":      ["S4"] + CICLICAS + LOCALES + GLOBALES,
    "D_solo_ROTI":     ["S4"] + CICLICAS + ["ROTI"],
}

# Periodo nativo de cada índice y retardo de disponibilidad (en múltiplos del periodo).
PERIODO_INDICES = {"Dst_Index": "1h", "AE_Index": "1h", "Kp_Index": "3h", "f10.7_Index": "1D"}


def agregar_ciclicas(df: pd.DataFrame) -> pd.DataFrame:
    """Periodicidad diaria y anual (deterministas, no requieren ajuste)."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    minuto = idx.hour * 60 + idx.minute
    out["daily_sin"] = np.sin(2 * np.pi * minuto / 1440)
    out["daily_cos"] = np.cos(2 * np.pi * minuto / 1440)
    dias = np.where(idx.is_leap_year, 366, 365)
    out["day_of_year_sin"] = np.sin(2 * np.pi * (idx.dayofyear - 1) / dias)
    out["day_of_year_cos"] = np.cos(2 * np.pi * (idx.dayofyear - 1) / dias)
    return out


def _valor_en_nodos(s: pd.Series, periodo: str) -> pd.Series:
    """Valor del índice al INICIO de cada periodo.

    Si el minuto exacto falta (p. ej. 00:00, ausente todos los días en el CSV
    real por el hueco diario 23:46–00:00), se recupera extrapolando hacia atrás
    con los dos primeros minutos disponibles del periodo. Esto es EXACTO tanto
    si la serie es escalonada (Kp, F10.7: pendiente 0) como si fue interpolada
    linealmente entre nodos (Dst, AE)."""
    inicio = s.index.floor(periodo)
    g = pd.DataFrame({"v": s.to_numpy(), "t": s.index, "ini": inicio})
    primero = g.groupby("ini").nth(0).set_index("ini")
    segundo = g.groupby("ini").nth(1).set_index("ini")
    dt0 = (primero["t"] - primero.index).dt.total_seconds() / 60
    dt1 = (segundo["t"].reindex(primero.index) - primero["t"]).dt.total_seconds() / 60
    pend = (segundo["v"].reindex(primero.index) - primero["v"]) / dt1
    v0 = primero["v"] - pend.fillna(0) * dt0
    v0[dt0 == 0] = primero["v"][dt0 == 0]
    return v0


def causalizar_indice(serie_minutal: pd.Series, periodo: str, retardo_periodos: int = 1) -> pd.Series:
    """Convierte un índice interpolado a minuto en su versión causal.

    1. Recupera el valor del índice al inicio de cada periodo (nodo).
    2. Lo desplaza `retardo_periodos` periodos: el promedio de [h, h+1) se usa desde h+1.
    3. Lo mantiene constante (forward-fill) hasta el siguiente valor disponible.

    En operación real puede existir latencia adicional de publicación (Dst/AE
    quick-look); se modela con `retardo_periodos` > 1.
    """
    s = serie_minutal.dropna().sort_index()
    disp = _valor_en_nodos(s, periodo)
    disp.index = disp.index + retardo_periodos * pd.Timedelta(periodo)
    idx = serie_minutal.index
    return disp.reindex(idx.union(disp.index)).ffill().reindex(idx)


def nodos_reales(serie: pd.Series, tol=1e-6, sampling_min=1) -> pd.Series:
    """True en los minutos que son mediciones reales; False en los minutos
    rellenados por interpolación lineal (interior de un tramo recto).

    Un minuto es "nodo" si la pendiente cambia justo después de él, si está en
    el borde de un hueco de muestreo, o si es el primero/último de la serie."""
    v = serie.to_numpy(dtype=float)
    t = serie.index
    contiguo = np.r_[False, np.diff(t.values) == np.timedelta64(sampling_min, "m")]
    d1 = np.r_[np.nan, np.diff(v)]
    d1[~contiguo] = np.nan
    cambia_despues = np.r_[np.abs(np.diff(d1)) > tol, True]
    borde = ~contiguo | ~np.r_[contiguo[1:], False] | np.isnan(np.r_[d1[1:], np.nan])
    return pd.Series(cambia_despues | borde | np.isnan(cambia_despues.astype(float)), index=t)


def diagnosticar_interpolacion(serie: pd.Series, **kw) -> dict:
    """Porcentaje de minutos que NO son medición real (relleno lineal)."""
    nodo = nodos_reales(serie.dropna(), **kw)
    return dict(pct_minutos_lineales=round(100 * (1 - nodo.mean()), 2), nodos=nodo)


def causalizar_tramos_lineales(serie: pd.Series, nodos: pd.Series) -> pd.Series:
    """Conserva solo las mediciones reales y las mantiene (forward-fill) hasta la siguiente:
    en el minuto t se usa el último valor que ya existía en t."""
    out = serie.copy()
    out[~nodos.reindex(out.index, fill_value=True).to_numpy()] = np.nan
    return out.ffill()


def construir_dataset(s4: pd.DataFrame, multivariable: pd.DataFrame | None = None,
                      causal: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Une el S4 reconstruido con los precursores (inner join por minuto) y
    agrega la codificación cíclica. El objetivo S4 es SIEMPRE el reconstruido."""
    df = s4.copy()
    if multivariable is not None:
        mv = multivariable.copy()
        if causal:
            for col, periodo in PERIODO_INDICES.items():
                if col in mv.columns:
                    mv[col] = causalizar_indice(mv[col], periodo)
            for col in LOCALES:
                if col in mv.columns:
                    diag = diagnosticar_interpolacion(mv[col])
                    if verbose:
                        print(f"  {col}: {diag['pct_minutos_lineales']} % de minutos rellenados linealmente -> se mantiene el último valor real")
                    mv[col] = causalizar_tramos_lineales(mv[col], diag["nodos"])
        df = df.join(mv, how="inner")
        antes = len(df)
        if verbose:
            nan = df.isna().mean().mul(100).round(2)
            print("  % NaN por columna tras causalizar:", nan[nan > 0].to_dict() or "ninguno")
        df = df.dropna()
        if verbose:
            print(f"Join S4 + precursores: {antes:,} filas, {antes - len(df):,} descartadas por NaN.")
        if len(df) < 0.9 * antes:
            raise ValueError("Se descartó más del 10 % de las filas por NaN: revisar la causalización "
                             "(ver '% NaN por columna' arriba).")
    return agregar_ciclicas(df)