"""Orquestación: de la configuración a particiones, corridas y registro.

Regla de oro aplicada aquí: TEST solo se evalúa cuando `evaluar_test=True`,
y eso se hace UNA vez, con la configuración final ya elegida en Validación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as ft
from . import io, splits
from .config import ruta
from .models import ajustar_climatologia, climatologia_diaria, persistencia
from .registry import registrar
from .windows import ajustar_escalador, preparar_particion


def preparar_datos(cfg, usar_precursores=True, verbose=True) -> dict:
    """Carga, une, recorta, particiona y audita. Devuelve todo lo necesario
    para construir particiones con cualquier conjunto de features."""
    d = cfg["datos"]
    s4 = io.cargar_s4_reconstruido(ruta(cfg, "s4_reconstruido"), d["flags_reconstruccion"])
    mv = None
    if usar_precursores and cfg["rutas"].get("multivariable"):
        mv = io.cargar_multivariable(ruta(cfg, "multivariable"))
    df = ft.construir_dataset(s4, mv, causal=True, verbose=verbose)
    df = io.recortar_periodo(df, cfg["periodo"]["inicio"], cfg["periodo"]["fin"])

    hc = cfg["split"].get("hora_corte_ut", 12)
    eventos = splits.identificar_eventos(df, d["target"], d["umbral_evento"], d["sampling_min"])
    sp = cfg["split"]
    if sp.get("fin_train") and sp.get("fin_validacion"):
        corte = dict(fin_train=sp["fin_train"], fin_validacion=sp["fin_validacion"], ranking=None)
    else:
        corte = splits.buscar_corte_temporal(df, eventos, d["target"], d["umbral_evento"], sp["ratios"],
                                             sp["min_eventos"], sp["min_dias_activos_test"], hc)
    train, val, test = splits.particionar(df, corte["fin_train"], corte["fin_validacion"], hc)
    splits.auditar_fuga(train, val, test, eventos)
    reporte = splits.reporte_particion(train, val, test, eventos, d["target"], d["umbral_evento"], hc)
    if verbose:
        print(reporte.to_string(index=False))
    return dict(df=df, eventos=eventos, corte=corte, train=train, val=val, test=test, reporte=reporte)


def construir_particiones(cfg, datos, features, lookback=None, horizon=None, starts=None) -> dict:
    """Escalador ajustado SOLO con Train sobre las features pedidas + particiones."""
    L = lookback or cfg["ventanas"]["lookback"]
    H = horizon or cfg["ventanas"]["horizon"]
    d = cfg["datos"]
    hc = cfg["split"].get("hora_corte_ut", 12)
    cols = list(dict.fromkeys([d["target"]] + list(features)))
    esc = ajustar_escalador(datos["train"], cols, cfg["escalador"])
    st = starts or {}
    parts = {k: preparar_particion(k, datos[k], esc, features, L, H, d["target"], d["sampling_min"], hc, st.get(k))
             for k in ("train", "val", "test")}
    parts["escalador"] = esc
    return parts


def baselines(cfg, datos, parts, particion="val") -> dict:
    """Predicciones de persistencia y climatología (escala física) para una partición."""
    p = parts[particion]
    perfil = ajustar_climatologia(datos["train"], cfg["datos"]["target"])
    return {"Persistencia": persistencia(p), "Climatología diaria": climatologia_diaria(p, perfil)}


def evaluar_y_registrar(cfg, p, yhat, nombre, grupo, semilla=None, umbral_alerta=None, extra=None) -> dict:
    d = cfg["datos"]
    ua = umbral_alerta if umbral_alerta is not None else d["umbral_evento"]
    fila = ev.evaluar(p, yhat, nombre, d["umbral_evento"], ua)
    ruta_reg = ruta(cfg, "experimentos") / f"registro_{cfg['nombre']}.csv"
    registrar(ruta_reg, cfg, fila, grupo, semilla, extra)
    return fila


def calibrar_umbral(cfg, p_val, yhat_val, criterio="HSS") -> tuple[float, pd.DataFrame]:
    """Umbral de alerta sobre la predicción, elegido en VALIDACIÓN."""
    m = ~p_val.objetivo_reconstruido
    tabla = ev.barrido_umbral_alerta(p_val.y_real[m], yhat_val[m], cfg["evaluacion"]["umbrales_alerta"],
                                     cfg["datos"]["umbral_evento"])
    return ev.elegir_umbral_alerta(tabla, criterio), tabla


def correr_variante(cfg, datos, features, nombre, grupo, semillas=None, particion_eval="val",
                    starts=None, guardar_modelo=False, verbose=0) -> list[dict]:
    """Entrena `nombre` con cada semilla y lo evalúa en `particion_eval` (por defecto VALIDACIÓN).
    Devuelve las filas registradas. Es la unidad de todos los experimentos de la tesis."""
    from .train import entrenar, predecir
    semillas = semillas or cfg["entrenamiento"]["semillas"]
    parts = construir_particiones(cfg, datos, features, starts=starts)
    filas, modelos = [], []
    for s in semillas:
        modelo, hist, seg = entrenar(cfg, parts["train"], parts["val"], parts["escalador"], s, verbose=verbose)
        p = parts[particion_eval]
        yhat = predecir(modelo, p, parts["escalador"], cfg["datos"]["target"])
        fila = evaluar_y_registrar(cfg, p, yhat, nombre, grupo, s,
                                   extra=dict(epocas=len(hist["loss"]), seg_entrenamiento=round(seg, 1),
                                              features="|".join(features)))
        filas.append(fila)
        print(f"  {nombre} | semilla {s} | {particion_eval}: RMSE_global={fila['RMSE_global']:.4f} "
              f"RMSE_evento={fila['RMSE_evento']:.4f} HSS={fila['HSS']:.3f} POD={fila['POD']:.3f} "
              f"Prec={fila['Precision']:.3f}")
        if guardar_modelo:
            modelos.append((s, modelo, yhat))
    return (filas, modelos, parts) if guardar_modelo else filas
