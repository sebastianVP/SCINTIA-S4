"""Entrenamiento y predicción genéricos (LSTM o PatchTST) + medición de latencia."""
from __future__ import annotations

import time

import numpy as np

from .losses import construir_perdida
from .models import construir_modelo
from .windows import a_escala_modelo, a_escala_real


def configurar_gpu(memoria_dinamica=True, determinista=False):
    """RTX 4080: memoria bajo demanda. `determinista=True` reduce el ruido entre
    corridas (en la tesis se observó variación de la arquitectura ganadora por
    no-determinismo de cuDNN) a costa de algo de velocidad."""
    import os
    import tensorflow as tf
    if determinista:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices("GPU"):
        if memoria_dinamica:
            tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", [g.name for g in tf.config.list_physical_devices("GPU")] or "ninguna (CPU)")


def entrenar(cfg, p_train, p_val, esc, semilla=42, epochs=None, verbose=2):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tr = cfg["entrenamiento"]
    umbral_esc = float(a_escala_modelo(cfg["datos"]["umbral_evento"], esc, cfg["datos"]["target"]))
    perdida = construir_perdida(cfg["perdida"], umbral_esc)
    cfg_modelo = dict(cfg["modelo"])
    if cfg_modelo["tipo"] == "patchtst":
        cfg_modelo["idx_target"] = p_train.features.index(cfg["datos"]["target"])
    modelo = construir_modelo(cfg_modelo, (p_train.lookback, len(p_train.features)), p_train.horizon,
                              perdida, tr["learning_rate"], seed=semilla)
    cb = [EarlyStopping(monitor="val_loss", patience=tr["patience_es"], restore_best_weights=True),
          ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=tr["patience_lr"], min_lr=1e-5)]
    t0 = time.time()
    hist = modelo.fit(p_train.dataset(tr["batch_size"], shuffle=True, seed=semilla),
                      validation_data=p_val.dataset(tr["batch_size"] * 2),
                      epochs=epochs or tr["epochs"], callbacks=cb, verbose=verbose)
    return modelo, hist.history, time.time() - t0


def predecir(modelo, p, esc, target="S4", batch_size=1024) -> np.ndarray:
    """Predicción en escala FÍSICA (sin abs(): un S4 negativo es un error del
    modelo que debe verse en las métricas, no ocultarse)."""
    y_esc = modelo.predict(p.dataset(batch_size), verbose=0)
    return a_escala_real(y_esc, esc, target)


def medir_latencia(modelo, p, n=200, dispositivo="/CPU:0") -> dict:
    """Latencia de UNA ventana (caso operativo), en ms. Se mide en CPU por defecto
    (despliegue Edge) y en GPU con dispositivo='/GPU:0'."""
    import tensorflow as tf
    x = p.X_lote(np.arange(min(n, p.n)))
    with tf.device(dispositivo):
        modelo(x[:1], training=False)  # calentamiento
        tiempos = []
        for i in range(len(x)):
            t0 = time.perf_counter()
            modelo(x[i:i + 1], training=False)
            tiempos.append(1000 * (time.perf_counter() - t0))
    return dict(dispositivo=dispositivo, lat_ms_mediana=float(np.median(tiempos)),
                lat_ms_p95=float(np.percentile(tiempos, 95)), parametros=int(modelo.count_params()))
