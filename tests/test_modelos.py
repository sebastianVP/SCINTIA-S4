import numpy as np

from s4 import features as ft
from s4.config import variante
from s4.pipeline import construir_particiones, preparar_datos
from s4.train import entrenar, medir_latencia, predecir


def _corrida(cfg, tipo, conjunto):
    cfg = variante(cfg, **{"modelo.tipo": tipo})
    d = preparar_datos(cfg, verbose=False)
    parts = construir_particiones(cfg, d, ft.CONJUNTOS[conjunto])
    m, hist, seg = entrenar(cfg, parts["train"], parts["val"], parts["escalador"], semilla=1, verbose=0)
    yhat = predecir(m, parts["val"], parts["escalador"])
    assert yhat.shape == parts["val"].y_real.shape and np.isfinite(yhat).all()
    lat = medir_latencia(m, parts["val"], n=5)
    assert lat["lat_ms_mediana"] > 0
    return yhat


def test_lstm_apilado(entorno):
    _corrida(entorno["cfg"], "apilado", "B_S4_ciclica")


def test_patchtst(entorno):
    _corrida(entorno["cfg"], "patchtst", "E_completo")
