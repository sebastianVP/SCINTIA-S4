"""Modelos de referencia sin aprendizaje profundo (cotas inferiores)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistencia(particion) -> np.ndarray:
    """Ŝ4(t+h) = S4(t) para h = 1..H, en escala física. Pedido explícito del especialista."""
    return np.repeat(particion.ultimo_real[:, None], particion.horizon, axis=1)


def ajustar_climatologia(train_df: pd.DataFrame, target="S4") -> pd.Series:
    """Perfil medio de S4 por minuto del día, SOLO con Train."""
    m = train_df.index.hour * 60 + train_df.index.minute
    return train_df[target].groupby(m).mean()


def climatologia_diaria(particion, perfil: pd.Series) -> np.ndarray:
    """Ŝ4(t+h) = media histórica de S4 a esa hora del día. Segundo baseline ingenuo:
    captura el ciclo día/noche sin mirar el estado actual de la ionosfera."""
    t = particion.t
    minutos = ((t.hour * 60 + t.minute).to_numpy()[:, None] + np.arange(1, particion.horizon + 1)[None, :]) % 1440
    return perfil.reindex(range(1440)).interpolate().to_numpy()[minutos]
