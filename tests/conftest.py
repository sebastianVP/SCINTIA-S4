import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def serie_sintetica(dias=60, seed=0):
    """S4 a 1 min con fondo diurno, eventos nocturnos (00–06 UT) y algunos huecos."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-26", periods=dias * 1440, freq="1min")
    hora = idx.hour + idx.minute / 60
    s4 = 0.12 + 0.03 * rng.standard_normal(len(idx)).cumsum() * 0.01 + 0.02 * rng.random(len(idx))
    for d in range(dias):
        if rng.random() < 0.35:
            ini = d * 1440 + int(rng.integers(30, 300))
            dur = int(rng.integers(20, 120))
            forma = np.sin(np.linspace(0, np.pi, dur)) * rng.uniform(0.6, 1.1)
            s4[ini:ini + dur] += forma
    s4 = np.clip(s4, 0.03, 1.5)
    df = pd.DataFrame({"S4": s4}, index=idx)
    df["Fix1"] = False
    df.iloc[rng.choice(len(df), 300, replace=False), df.columns.get_loc("Fix1")] = True
    huecos = rng.choice(np.arange(1440, len(df) - 1440), 5, replace=False)
    quitar = np.concatenate([np.arange(h, h + 30) for h in huecos])
    df = df.drop(df.index[quitar])
    _ = hora
    return df


def multivariable_sintetico(idx, seed=1):
    rng = np.random.default_rng(seed)
    horas = pd.date_range(idx.min().floor("1h"), idx.max().ceil("1h") + pd.Timedelta(hours=1), freq="1h")
    dst_h = pd.Series(rng.integers(-60, 10, len(horas)).astype(float), index=horas)
    ae_h = pd.Series(rng.integers(20, 600, len(horas)).astype(float), index=horas)
    full = pd.date_range(horas.min(), horas.max(), freq="1min")
    dst = dst_h.reindex(full).interpolate()     # interpolación lineal = fuga (como el CSV real)
    ae = ae_h.reindex(full).interpolate()
    kp = pd.Series(rng.integers(0, 6, len(full) // 180 + 2).astype(float).repeat(180)[:len(full)], index=full)
    f107 = pd.Series(150.0 + rng.normal(0, 5, len(full) // 1440 + 2).repeat(1440)[:len(full)], index=full)
    tec = pd.Series(40 + 20 * np.sin(np.arange(len(full)) / 1440 * 2 * np.pi) + rng.normal(0, 1, len(full)), index=full)
    df = pd.DataFrame({"TEC": tec, "ROTEC": tec.diff().fillna(0), "ROTI": tec.diff().rolling(5).std().bfill(),
                       "Kp_Index": kp, "Dst_Index": dst, "AE_Index": ae, "f10.7_Index": f107})
    return df.reindex(idx), dst_h


@pytest.fixture(scope="session")
def entorno(tmp_path_factory):
    base = tmp_path_factory.mktemp("datos")
    s4 = serie_sintetica()
    s4.reset_index(names="Tiempo").to_csv(base / "s4.csv", index=False)
    mv, dst_h = multivariable_sintetico(s4.index)
    mv.reset_index(names="Tiempo").to_csv(base / "mv.csv", index=False)
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config/ablacion_2025_2026.yaml"))
    cfg["rutas"].update(base=str(base), s4_reconstruido="s4.csv", multivariable="mv.csv")
    cfg["periodo"].update(inicio="2025-01-26", fin="2025-03-26")
    cfg["ventanas"].update(lookback=60, horizon=10)
    cfg["entrenamiento"].update(epochs=1, batch_size=128)
    return dict(cfg=cfg, s4=s4, mv=mv, dst_h=dst_h, base=base)
