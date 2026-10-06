"""Pruebas estadísticas para comparar dos modelos sobre el mismo Test.

Por qué cambia respecto al notebook anterior
--------------------------------------------
Allí el bootstrap remuestreaba VENTANAS individuales. Dos ventanas consecutivas
comparten LOOKBACK-1 minutos de entrada y H-1 minutos de objetivo: no son
independientes, y remuestrearlas como si lo fueran estrecha artificialmente el
intervalo de confianza. Aquí la unidad de remuestreo es el DÍA IONOSFÉRICO
(bootstrap por bloques), que contiene una noche completa de actividad.

Se añade la prueba de Diebold–Mariano (con corrección de Harvey–Leybourne–
Newbold) sobre la diferencia diaria de pérdidas, estándar en la literatura de
pronóstico para comparar dos modelos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def _agrupar_por_dia(dia):
    codigos, uniq = pd.factorize(pd.DatetimeIndex(dia))
    return codigos, len(uniq)


def bootstrap_bloques(y, yhat_a, yhat_b, dia, metrica, n_boot=1000, seed=42, mascara=None) -> dict:
    """IC 95 % de metrica(A) - metrica(B), remuestreando días completos.

    metrica: función (y, yhat) -> float, p. ej.
        lambda y, yh: rmse sobre y >= 0.6        (RMSE en eventos)
        lambda y, yh: HSS de la alerta           (habilidad de alerta)
    Diferencia > 0 con IC que no cruza 0  =>  B es significativamente MEJOR (si menor es mejor).
    """
    if mascara is not None:
        y, yhat_a, yhat_b, dia = y[mascara], yhat_a[mascara], yhat_b[mascara], np.asarray(dia)[mascara]
    cod, nd = _agrupar_por_dia(dia)
    orden = np.argsort(cod, kind="stable")
    limites = np.r_[0, np.cumsum(np.bincount(cod, minlength=nd))]
    rng = np.random.default_rng(seed)
    base = metrica(y, yhat_a) - metrica(y, yhat_b)
    difs = []
    for _ in range(n_boot):
        dias = rng.integers(0, nd, nd)
        idx = np.concatenate([orden[limites[d]:limites[d + 1]] for d in dias])
        try:
            d = metrica(y[idx], yhat_a[idx]) - metrica(y[idx], yhat_b[idx])
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(d):
            difs.append(d)
    difs = np.array(difs)
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return dict(diferencia=float(base), ic95=(float(lo), float(hi)), n_dias=nd, n_boot_validos=len(difs),
                significativo=bool(lo > 0 or hi < 0))


def diebold_mariano(y, yhat_a, yhat_b, dia, potencia=2, mascara=None) -> dict:
    """DM sobre la pérdida media DIARIA d_k = L(A)_k - L(B)_k.
    d > 0 en promedio  =>  B tiene menor pérdida que A."""
    if mascara is not None:
        y, yhat_a, yhat_b, dia = y[mascara], yhat_a[mascara], yhat_b[mascara], np.asarray(dia)[mascara]
    la = np.mean(np.abs(y - yhat_a) ** potencia, axis=1)
    lb = np.mean(np.abs(y - yhat_b) ** potencia, axis=1)
    d = pd.Series(la - lb).groupby(pd.DatetimeIndex(dia)).mean().to_numpy()
    n = len(d)
    dbar = d.mean()
    # Varianza de largo plazo (Newey–West, rezago 1 día)
    g0 = np.var(d, ddof=0)
    g1 = np.cov(d[1:], d[:-1], ddof=0)[0, 1] if n > 2 else 0.0
    var = (g0 + 2 * g1) / n
    dm = dbar / np.sqrt(var) if var > 0 else np.nan
    h = 2  # horizonte efectivo en días del rezago usado
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * (1 - st.t.cdf(abs(stat), df=n - 1))
    return dict(DM=float(stat), p_valor=float(p), n_dias=n, media_dif=float(dbar))


# Métricas listas para usar con bootstrap_bloques -----------------------------
def m_rmse_evento(umbral=0.6):
    def f(y, yh):
        ev = y >= umbral
        return float(np.sqrt(np.mean((y[ev] - yh[ev]) ** 2))) if ev.any() else np.nan
    return f


def m_rmse_global(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def m_menos_hss(umbral=0.6, umbral_alerta=0.6):
    """-HSS (para que 'menor es mejor' igual que el RMSE)."""
    from .evaluate import metricas_alerta
    return lambda y, yh: -metricas_alerta(y, yh, umbral, umbral_alerta)["HSS"]
