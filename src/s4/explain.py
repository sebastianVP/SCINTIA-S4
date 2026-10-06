"""Explicabilidad: ¿qué variables usa realmente el modelo? (pregunta sobre ROTI)

1. Importancia por permutación (agnóstica al modelo): se permuta una variable
   ENTRE ventanas (rompe su relación con el objetivo, conserva su distribución)
   y se mide cuánto empeora la métrica. Se calcula sobre VALIDACIÓN.
2. Gradientes integrados (Sundararajan et al., 2017): atribución por variable y
   por minuto de la ventana -> mapa de saliencia (variable × retardo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import a_escala_real


def importancia_permutacion(modelo, p, esc, metrica, n_ventanas=20000, repeticiones=3, seed=42,
                            solo_eventos=False, umbral=0.6, target="S4") -> pd.DataFrame:
    """metrica(y_real, yhat_real) -> float, menor es mejor (p. ej. RMSE en eventos)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(p.n)
    if solo_eventos:
        idx = idx[p.y_real.max(axis=1) >= umbral]
    idx = rng.choice(idx, min(n_ventanas, len(idx)), replace=False)
    X = p.X_lote(idx)
    y = p.y_real[idx]
    base = metrica(y, a_escala_real(modelo.predict(X, verbose=0, batch_size=1024), esc, target))
    filas = []
    for j, f in enumerate(p.features):
        deltas = []
        for _ in range(repeticiones):
            Xp = X.copy()
            Xp[:, :, j] = Xp[rng.permutation(len(X)), :, j]
            deltas.append(metrica(y, a_escala_real(modelo.predict(Xp, verbose=0, batch_size=1024), esc, target)) - base)
        filas.append(dict(variable=f, aumento_error=np.mean(deltas), std=np.std(deltas)))
    return pd.DataFrame(filas).sort_values("aumento_error", ascending=False).reset_index(drop=True)


def gradientes_integrados(modelo, X, pasos=32, salida=None) -> np.ndarray:
    """Atribuciones (n, L, F) respecto de la suma de las H salidas (o de una salida h).
    Línea base: ventana de ceros (mínimo de la escala MinMax)."""
    import tensorflow as tf
    X = tf.convert_to_tensor(X, tf.float32)
    base = tf.zeros_like(X)
    alphas = tf.linspace(0.0, 1.0, pasos + 1)
    grads = []
    for a in alphas:
        xi = base + a * (X - base)
        with tf.GradientTape() as tape:
            tape.watch(xi)
            out = modelo(xi, training=False)
            obj = tf.reduce_sum(out if salida is None else out[:, salida], axis=-1)
        grads.append(tape.gradient(obj, xi))
    g = tf.stack(grads)
    prom = (g[:-1] + g[1:]) / 2.0
    return ((X - base) * tf.reduce_mean(prom, axis=0)).numpy()


def resumen_saliencia(atrib: np.ndarray, features) -> pd.DataFrame:
    """Importancia media |atribución| por variable (normalizada a 100 %)."""
    imp = np.abs(atrib).mean(axis=(0, 1))
    return (pd.DataFrame({"variable": features, "importancia_pct": 100 * imp / imp.sum()})
            .sort_values("importancia_pct", ascending=False).reset_index(drop=True))
