"""Evaluador ÚNICO: todo modelo (persistencia, LSTM, PatchTST) se mide con estas
funciones, sobre la misma partición y en escala física.

Tres familias de métricas:
1. Regresión: RMSE/MAE global, en minutos de evento y por paso del horizonte.
   NOTA: el RMSE "en eventos" (solo y_real >= 0.6) premia sobreestimar; nunca
   debe usarse solo para seleccionar modelos -> se reporta junto a 2.
2. Alerta (lo operativo): una ventana dispara alerta si max(ŷ) >= umbral_alerta;
   hay evento si max(y) >= 0.6. Tabla de contingencia -> POD (Recall),
   Precisión, FAR (= 1 - Precisión), POFD, CSI, HSS, F1.
   El umbral de alerta sobre la PREDICCIÓN se calibra en Validación (puede
   diferir de 0.6) y se aplica sin cambios en Test.
3. Fase del evento: separa ventanas de INICIO (S4(t) < 0.6 y llega un evento),
   SOSTENIDAS (S4(t) >= 0.6 y sigue), FIN y CALMA. El valor de un sistema de
   alerta temprana está en el INICIO, que la persistencia no puede anticipar.

Los minutos reconstruidos (Fix0–Fix4) se excluyen por defecto (`excluir_reconstruidos`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else np.nan


def _mae(a, b):
    return float(np.mean(np.abs(a - b))) if a.size else np.nan


def _mascara(p, excluir_reconstruidos):
    return ~p.objetivo_reconstruido if excluir_reconstruidos else np.ones(p.n, bool)


# -----------------------------------------------------------------------------
# 1. Regresión
# -----------------------------------------------------------------------------
def metricas_regresion(y, yhat, umbral=0.6) -> dict:
    ev = y >= umbral
    r = dict(RMSE_global=_rmse(y, yhat), MAE_global=_mae(y, yhat),
             RMSE_evento=_rmse(y[ev], yhat[ev]), MAE_evento=_mae(y[ev], yhat[ev]),
             RMSE_calma=_rmse(y[~ev], yhat[~ev]),
             sesgo_evento=float(np.mean(yhat[ev] - y[ev])) if ev.any() else np.nan,
             RMSE_tH=_rmse(y[:, -1], yhat[:, -1]),
             n_min_evento=int(ev.sum()))
    return r


def rmse_por_paso(y, yhat, umbral=0.6) -> pd.DataFrame:
    filas = []
    for h in range(y.shape[1]):
        ev = y[:, h] >= umbral
        filas.append(dict(h=h + 1, RMSE=_rmse(y[:, h], yhat[:, h]), RMSE_evento=_rmse(y[ev, h], yhat[ev, h])))
    return pd.DataFrame(filas)


# -----------------------------------------------------------------------------
# 2. Alerta
# -----------------------------------------------------------------------------
def contingencia(evento: np.ndarray, alerta: np.ndarray) -> dict:
    tp = int(np.sum(evento & alerta)); fp = int(np.sum(~evento & alerta))
    fn = int(np.sum(evento & ~alerta)); tn = int(np.sum(~evento & ~alerta))
    n = tp + fp + fn + tn
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    pofd = fp / (fp + tn) if fp + tn else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    esperado = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n if n else np.nan
    hss = ((tp + tn) - esperado) / (n - esperado) if n and n != esperado else np.nan
    f1 = 2 * prec * pod / (prec + pod) if prec and pod and not np.isnan(prec + pod) else np.nan
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, POD=pod, Precision=prec, FAR=1 - prec if prec == prec else np.nan,
                POFD=pofd, CSI=csi, HSS=hss, F1=f1)


def metricas_alerta(y, yhat, umbral_evento=0.6, umbral_alerta=0.6) -> dict:
    """Nivel ventana: ¿habrá S4 >= 0.6 en los próximos H minutos?"""
    return contingencia(y.max(axis=1) >= umbral_evento, yhat.max(axis=1) >= umbral_alerta)


def barrido_umbral_alerta(y, yhat, umbrales, umbral_evento=0.6) -> pd.DataFrame:
    """Curva operativa; se calcula en VALIDACIÓN para elegir el umbral de alerta."""
    return pd.DataFrame([{"umbral_alerta": u, **metricas_alerta(y, yhat, umbral_evento, u)} for u in umbrales])


def elegir_umbral_alerta(tabla_val: pd.DataFrame, criterio="HSS") -> float:
    return float(tabla_val.loc[tabla_val[criterio].idxmax(), "umbral_alerta"])


# -----------------------------------------------------------------------------
# 3. Fases del evento
# -----------------------------------------------------------------------------
def fase_ventana(ultimo, y, umbral=0.6) -> np.ndarray:
    ahora = ultimo >= umbral
    futuro = y.max(axis=1) >= umbral
    fase = np.full(len(ultimo), "calma", dtype=object)
    fase[~ahora & futuro] = "inicio"
    fase[ahora & futuro] = "sostenido"
    fase[ahora & ~futuro] = "fin"
    return fase


def metricas_por_fase(p, yhat, umbral=0.6, umbral_alerta=0.6, excluir_reconstruidos=True) -> pd.DataFrame:
    m = _mascara(p, excluir_reconstruidos)
    y, yh, ult = p.y_real[m], yhat[m], p.ultimo_real[m]
    fase = fase_ventana(ult, y, umbral)
    filas = []
    for f in ("inicio", "sostenido", "fin", "calma"):
        s = fase == f
        if not s.any():
            continue
        alerta = yh[s].max(axis=1) >= umbral_alerta
        filas.append(dict(fase=f, n_ventanas=int(s.sum()), RMSE=_rmse(y[s], yh[s]),
                          tasa_alerta=float(alerta.mean())))
    return pd.DataFrame(filas)


def anticipacion_inicio(p, yhat, umbral=0.6, umbral_alerta=0.6, excluir_reconstruidos=True) -> dict:
    """Para ventanas de INICIO: ¿con cuántos minutos de antelación se alertó?
    Antelación = minutos entre t y el primer minuto real >= 0.6 (solo si hubo alerta)."""
    m = _mascara(p, excluir_reconstruidos)
    y, yh, ult = p.y_real[m], yhat[m], p.ultimo_real[m]
    ini = fase_ventana(ult, y, umbral) == "inicio"
    if not ini.any():
        return dict(n_inicio=0)
    primer = np.argmax(y[ini] >= umbral, axis=1) + 1
    alerta = yh[ini].max(axis=1) >= umbral_alerta
    return dict(n_inicio=int(ini.sum()), POD_inicio=float(alerta.mean()),
                antelacion_media_min=float(primer[alerta].mean()) if alerta.any() else np.nan)


# -----------------------------------------------------------------------------
# Falsas alarmas: ¿son "casi eventos"?
# -----------------------------------------------------------------------------
def analisis_falsas_alarmas(p, yhat, umbral=0.6, umbral_alerta=0.6, excluir_reconstruidos=True) -> pd.DataFrame:
    """Distribución del S4 REAL máximo en las ventanas con falsa alarma.
    Responde a la pregunta del especialista: ¿las falsas alarmas se agrupan en S4 ~ 0.55?"""
    m = _mascara(p, excluir_reconstruidos)
    y, yh = p.y_real[m], yhat[m]
    fa = (y.max(axis=1) < umbral) & (yh.max(axis=1) >= umbral_alerta)
    bins = [0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6]
    cat = pd.cut(y.max(axis=1)[fa], bins, right=False)
    t = cat.value_counts().sort_index().rename("n").to_frame()
    t["pct"] = (100 * t["n"] / max(1, fa.sum())).round(2)
    return t


# -----------------------------------------------------------------------------
# Resumen de una corrida
# -----------------------------------------------------------------------------
def evaluar(p, yhat, nombre, umbral=0.6, umbral_alerta=0.6, excluir_reconstruidos=True) -> dict:
    """Una fila con todo lo que va a las tablas de la tesis."""
    m = _mascara(p, excluir_reconstruidos)
    y, yh = p.y_real[m], yhat[m]
    fila = dict(modelo=nombre, particion=p.nombre, n_ventanas=int(m.sum()), umbral_alerta=umbral_alerta)
    fila.update(metricas_regresion(y, yh, umbral))
    fila.update(metricas_alerta(y, yh, umbral, umbral_alerta))
    fila.update(anticipacion_inicio(p, yhat, umbral, umbral_alerta, excluir_reconstruidos))
    return fila


def skill(metrica_modelo, metrica_referencia):
    """Skill score = 1 - modelo/referencia (> 0: el modelo mejora a la referencia)."""
    return 1 - metrica_modelo / metrica_referencia
