"""Partición cronológica Out-of-Time, identificación de eventos y auditoría de fuga.

Migrado de TESIS_MCC_01102026_0048.ipynb (secciones 4–8) con dos mejoras:

1. `hora_corte_ut`: el corte entre particiones se hace en un "día ionosférico"
   que empieza a esa hora UT. Con 0 (medianoche UT = 19 LT en Lima) una misma
   noche de centelleo puede quedar partida entre Train y Validación. Con 12
   (07 LT, sin centelleo) cada noche queda completa en una sola partición.
2. La auditoría detecta explícitamente eventos que CRUZAN un corte (la versión
   anterior solo verificaba que no estuvieran repetidos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Eventos
# -----------------------------------------------------------------------------
def identificar_eventos(df: pd.DataFrame, target="S4", umbral=0.6, sampling_min=1) -> pd.DataFrame:
    """Un evento = tramo CONSECUTIVO (sin huecos de muestreo) con S4 >= umbral."""
    s = df[target].to_numpy(dtype=float)
    idx = pd.DatetimeIndex(df.index)
    activo = s >= umbral
    cols = ["event_id", "start", "end", "duration_min", "n_observations", "max_S4"]
    if not activo.any():
        return pd.DataFrame(columns=cols)

    salto = np.r_[True, np.diff(idx.values) != np.timedelta64(sampling_min, "m")]
    cambia = np.r_[True, activo[1:] != activo[:-1]] | salto
    run_id = np.cumsum(cambia) - 1
    filas = []
    for rid in np.unique(run_id[activo]):
        pos = np.flatnonzero(run_id == rid)
        filas.append(dict(event_id=len(filas), start=idx[pos[0]], end=idx[pos[-1]],
                          duration_min=len(pos) * sampling_min, n_observations=len(pos),
                          max_S4=float(s[pos].max())))
    return pd.DataFrame(filas)


def dia_ionosferico(index, hora_corte_ut=12) -> pd.DatetimeIndex:
    """Fecha del 'día ionosférico' que empieza a `hora_corte_ut` UT."""
    return (pd.DatetimeIndex(index) - pd.Timedelta(hours=hora_corte_ut)).normalize()


# -----------------------------------------------------------------------------
# Búsqueda del corte
# -----------------------------------------------------------------------------
def buscar_corte_temporal(df, eventos, target="S4", umbral=0.6, ratios=(0.70, 0.15, 0.15),
                          min_eventos=(1, 1, 1), min_dias_activos_test=1, hora_corte_ut=12, top_n=10):
    """Evalúa cortes por día ionosférico cercanos a 70/15/15 y elige el que
    garantiza eventos en las 3 particiones con la actividad más pareja.

    La búsqueda usa solo conteos diarios (no entrena nada ni ajusta escalas), así
    que no introduce información de Validación/Test en el modelo."""
    dia = dia_ionosferico(df.index, hora_corte_ut)
    diario = (pd.DataFrame({"S4": df[target].to_numpy()}, index=dia)
              .groupby(level=0)["S4"].agg(observations="size", max_S4="max"))
    diario["active"] = diario["max_S4"] >= umbral
    cum_obs = diario["observations"].cumsum().to_numpy()
    cum_act = diario["active"].astype(int).cumsum().to_numpy()
    fechas = diario.index
    n = len(fechas)
    if n < 10:
        raise ValueError("Muy pocos días para una partición robusta.")

    ev_ini = dia_ionosferico(eventos["start"], hora_corte_ut) if len(eventos) else pd.DatetimeIndex([])
    ev_pos = np.searchsorted(fechas.values, ev_ini.values) if len(eventos) else np.array([], int)
    cum_ev = np.bincount(ev_pos, minlength=n).cumsum() if len(eventos) else np.zeros(n, int)

    def m(a, b):  # métricas del rango de días [a, b]
        g = lambda c: c[b] - (c[a - 1] if a > 0 else 0)
        dias = b - a + 1
        act = int(g(cum_act))
        return dict(days=dias, active_days=act, active_pct=100 * act / dias,
                    events=int(g(cum_ev)), observations=int(g(cum_obs)))

    tr, va, te = ratios
    total = cum_obs[-1]
    c_tr, c_va = round(n * tr), round(n * (tr + va))
    r = max(10, round(n * 0.10))
    act_global = 100 * diario["active"].mean()
    cand = []
    for i in range(max(1, c_tr - r), min(n - 2, c_tr + r) + 1):
        for j in range(max(i + 1, c_va - r), min(n - 1, c_va + r) + 1):
            mtr, mva, mte = m(0, i - 1), m(i, j - 1), m(j, n - 1)
            ratio_err = (abs(mtr["observations"] / total - tr) + abs(mva["observations"] / total - va)
                         + abs(mte["observations"] / total - te))
            ok = (mtr["events"] >= min_eventos[0] and mva["events"] >= min_eventos[1]
                  and mte["events"] >= min_eventos[2] and mte["active_days"] >= min_dias_activos_test)
            act_err = sum(abs(x["active_pct"] - act_global) for x in (mtr, mva, mte))
            cand.append(dict(fin_train=fechas[i - 1], fin_validacion=fechas[j - 1], factible=ok,
                             train=mtr, val=mva, test=mte, ratio_error=ratio_err, activity_error=act_err))
    fact = [c for c in cand if c["factible"]] or cand
    rank = sorted(fact, key=lambda c: (round(c["activity_error"], 1), c["ratio_error"]))
    tabla = pd.DataFrame([{
        "rank": k + 1, "fin_train": c["fin_train"].date(), "fin_validacion": c["fin_validacion"].date(),
        **{f"{p}_{q}": c[p][q] for p in ("train", "val", "test") for q in ("days", "active_days", "events")},
        "ratio_error": round(c["ratio_error"], 4), "activity_error": round(c["activity_error"], 2),
    } for k, c in enumerate(rank[:top_n])])
    elegido = rank[0]
    return dict(fin_train=elegido["fin_train"], fin_validacion=elegido["fin_validacion"],
                hora_corte_ut=hora_corte_ut, ranking=tabla, actividad_global_pct=act_global)


# -----------------------------------------------------------------------------
# Partición y auditoría
# -----------------------------------------------------------------------------
def particionar(df, fin_train, fin_validacion, hora_corte_ut=12):
    """TRAIN < VALIDACIÓN < TEST por día ionosférico completo, sin muestreo aleatorio."""
    fin_train, fin_validacion = pd.Timestamp(fin_train), pd.Timestamp(fin_validacion)
    if fin_train >= fin_validacion:
        raise ValueError("fin_train debe ser anterior a fin_validacion.")
    dia = dia_ionosferico(df.index, hora_corte_ut)
    train = df.loc[dia <= fin_train]
    val = df.loc[(dia > fin_train) & (dia <= fin_validacion)]
    test = df.loc[dia > fin_validacion]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Alguna partición quedó vacía.")
    assert len(train) + len(val) + len(test) == len(df)
    return train, val, test


def auditar_fuga(train, val, test, eventos=None) -> bool:
    """Lanza AssertionError ante cualquier violación; no continúa con una partición contaminada."""
    err = []
    for nombre, p in (("train", train), ("val", val), ("test", test)):
        if not p.index.is_monotonic_increasing:
            err.append(f"{nombre} no está ordenado.")
    if train.index.max() >= val.index.min():
        err.append("Validación no empieza después de Train.")
    if val.index.max() >= test.index.min():
        err.append("Test no empieza después de Validación.")
    if eventos is not None and len(eventos):
        cortes = [(val.index.min(), "train/val"), (test.index.min(), "val/test")]
        for t, nombre in cortes:
            cruzan = eventos[(eventos["start"] < t) & (eventos["end"] >= t)]
            if len(cruzan):
                err.append(f"{len(cruzan)} evento(s) cruzan el corte {nombre}.")
    if err:
        raise AssertionError("AUDITORÍA FALLIDA: " + " | ".join(err))
    print("✓ Auditoría de fuga: orden estricto TRAIN < VAL < TEST, sin solapamientos "
          "y ningún evento partido entre particiones.")
    return True


def reporte_particion(train, val, test, eventos, target="S4", umbral=0.6, hora_corte_ut=12) -> pd.DataFrame:
    filas = []
    for nombre, p in (("TRAIN", train), ("VALIDACIÓN", val), ("TEST", test)):
        dia = dia_ionosferico(p.index, hora_corte_ut)
        diario = pd.Series(p[target].to_numpy(), index=dia).groupby(level=0).max()
        ev = eventos[(eventos["start"] >= p.index.min()) & (eventos["end"] <= p.index.max())] if len(eventos) else eventos
        filas.append(dict(particion=nombre, inicio=p.index.min(), fin=p.index.max(), observaciones=len(p),
                          dias=len(diario), dias_activos=int((diario >= umbral).sum()),
                          pct_dias_activos=round(100 * (diario >= umbral).mean(), 2),
                          eventos=len(ev), minutos_evento=int((p[target] >= umbral).sum())))
    return pd.DataFrame(filas)
