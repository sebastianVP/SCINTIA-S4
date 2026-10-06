"""Escalado y ventanas X/Y sin materializar X (sin OOM), con metadatos para evaluar.

Migrado de TESIS_MCC_01102026_0048.ipynb (secciones 11–15). Novedades:
- `Particion` guarda, por ventana: instante t, día ionosférico, último S4
  observado (escala real), Y real y si algún minuto objetivo fue reconstruido.
- `inicios_comunes()` permite comparar distintos LOOKBACK sobre EXACTAMENTE
  los mismos instantes objetivo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .splits import dia_ionosferico


# -----------------------------------------------------------------------------
# Escalado (FIT solo en TRAIN)
# -----------------------------------------------------------------------------
def ajustar_escalador(train: pd.DataFrame, columnas, tipo="minmax"):
    esc = MinMaxScaler((0, 1)) if tipo == "minmax" else StandardScaler()
    esc.fit(train[columnas].astype(float))
    esc.columnas_ = list(columnas)
    return esc


def escalar(df: pd.DataFrame, esc) -> np.ndarray:
    return esc.transform(df[esc.columnas_].astype(float)).astype(np.float32)


def a_escala_real(y_esc: np.ndarray, esc, target="S4") -> np.ndarray:
    i = esc.columnas_.index(target)
    if isinstance(esc, MinMaxScaler):
        return y_esc * (esc.data_max_[i] - esc.data_min_[i]) + esc.data_min_[i]
    return y_esc * esc.scale_[i] + esc.mean_[i]


def a_escala_modelo(y_real, esc, target="S4"):
    i = esc.columnas_.index(target)
    if isinstance(esc, MinMaxScaler):
        return (y_real - esc.data_min_[i]) / (esc.data_max_[i] - esc.data_min_[i])
    return (y_real - esc.mean_[i]) / esc.scale_[i]


# -----------------------------------------------------------------------------
# Índices de ventanas válidas
# -----------------------------------------------------------------------------
def inicios_validos(index: pd.DatetimeIndex, lookback: int, horizon: int, sampling_min=1) -> np.ndarray:
    """Posición de inicio de cada ventana [start, start+lookback+horizon) sin huecos."""
    n = len(index)
    if n < lookback + horizon:
        return np.array([], dtype=np.int64)
    invalido = np.diff(index.values) != np.timedelta64(sampling_min, "m")
    prefijo = np.concatenate([[0], np.cumsum(invalido, dtype=np.int64)])
    starts = np.arange(0, n - lookback - horizon + 1)
    fin_obj = starts + lookback + horizon - 1
    return starts[(prefijo[fin_obj] - prefijo[starts]) == 0]


def inicios_comunes(index, lookbacks, horizon, sampling_min=1):
    """Para cada LOOKBACK, los inicios cuyo instante t coincide con los válidos
    del LOOKBACK más largo -> todas las variantes se evalúan en los mismos objetivos."""
    lmax = max(lookbacks)
    base = inicios_validos(index, lmax, horizon, sampling_min)
    return {L: base + (lmax - L) for L in lookbacks}


# -----------------------------------------------------------------------------
# Partición lista para entrenar/evaluar
# -----------------------------------------------------------------------------
@dataclass
class Particion:
    nombre: str
    X_full: np.ndarray            # (n_filas, n_features) escalado
    starts: np.ndarray            # inicios de ventana
    lookback: int
    horizon: int
    y: np.ndarray                 # (n_ventanas, H) escala del modelo
    y_real: np.ndarray            # (n_ventanas, H) escala física
    t: pd.DatetimeIndex           # instante t (último minuto observado)
    dia: pd.DatetimeIndex         # día ionosférico de t (bloques del bootstrap)
    ultimo_real: np.ndarray       # S4(t) en escala física
    objetivo_reconstruido: np.ndarray  # True si algún minuto de Y fue reconstruido
    features: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.starts)

    def dataset(self, batch_size=256, shuffle=False, seed=None):
        """tf.data que recorta X al vuelo (nunca materializa todas las ventanas)."""
        import tensorflow as tf
        feats = tf.constant(self.X_full)
        nf, L = self.X_full.shape[1], self.lookback
        ds = tf.data.Dataset.from_tensor_slices((self.starts.astype(np.int64), self.y))
        if shuffle:
            ds = ds.shuffle(min(self.n, 20_000), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda s, y: (tf.slice(feats, [s, 0], [L, nf]), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def X_lote(self, idx):
        """Materializa SOLO las ventanas pedidas (para explicabilidad o depuración)."""
        pos = self.starts[idx][:, None] + np.arange(self.lookback)[None, :]
        return self.X_full[pos]


def preparar_particion(nombre, df: pd.DataFrame, esc, features, lookback, horizon, target="S4",
                       sampling_min=1, hora_corte_ut=12, starts=None) -> Particion:
    X_full = escalar(df, esc)
    if starts is None:
        starts = inicios_validos(df.index, lookback, horizon, sampling_min)
    idx_t = esc.columnas_.index(target)
    cols = [esc.columnas_.index(c) for c in features]
    X_feat = X_full[:, cols]

    pos_y = starts[:, None] + lookback + np.arange(horizon)[None, :]
    y = X_full[:, idx_t][pos_y].astype(np.float32)
    s4_real = df[target].to_numpy(dtype=float)
    y_real = s4_real[pos_y]
    pos_t = starts + lookback - 1
    rec = df["reconstruido"].to_numpy() if "reconstruido" in df.columns else np.zeros(len(df), bool)
    t = df.index[pos_t]
    p = Particion(nombre, X_feat, starts, lookback, horizon, y, y_real, t,
                  dia_ionosferico(t, hora_corte_ut), s4_real[pos_t], rec[pos_y].any(axis=1), list(features))
    print(f"{nombre:10s}: {p.n:,} ventanas | X=({lookback},{len(features)}) | Y=({horizon}) | "
          f"objetivos reconstruidos: {p.objetivo_reconstruido.mean()*100:.2f} %")
    return p
