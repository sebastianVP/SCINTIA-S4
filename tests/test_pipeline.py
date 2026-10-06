import numpy as np
import pandas as pd
import pytest

from s4 import evaluate as ev, features as ft, splits, stats, windows
from s4.models import persistencia
from s4.pipeline import baselines, construir_particiones, preparar_datos


def test_causalidad_indices(entorno):
    """En el minuto t solo puede usarse el Dst de una hora YA cerrada."""
    mv, dst_h = entorno["mv"], entorno["dst_h"]
    caus = ft.causalizar_indice(mv["Dst_Index"].dropna(), "1h")
    t = pd.Timestamp("2025-02-03 10:35")
    assert caus.loc[t] == dst_h.loc[pd.Timestamp("2025-02-03 09:00")]
    # la serie interpolada original mezclaba valores de 10:00 y 11:00 (fuga)
    h10, h11 = dst_h.loc[pd.Timestamp("2025-02-03 10:00")], dst_h.loc[pd.Timestamp("2025-02-03 11:00")]
    if h10 != h11:
        assert mv["Dst_Index"].loc[t] not in (h10, caus.loc[t]) or h10 == caus.loc[t]


def test_split_sin_fuga(entorno):
    d = preparar_datos(entorno["cfg"], verbose=False)
    tr, va, te = d["train"], d["val"], d["test"]
    assert tr.index.max() < va.index.min() < te.index.min()
    # cortes a las 12 UT
    assert va.index.min().hour == 12 and te.index.min().hour == 12
    ev_ = d["eventos"]
    for t in (va.index.min(), te.index.min()):
        assert not ((ev_["start"] < t) & (ev_["end"] >= t)).any()


def test_auditoria_detecta_evento_partido():
    idx = pd.date_range("2025-01-01", periods=10, freq="1min")
    df = pd.DataFrame({"S4": [0.1, 0.1, 0.7, 0.7, 0.7, 0.7, 0.1, 0.1, 0.1, 0.1]}, index=idx)
    eventos = splits.identificar_eventos(df)
    with pytest.raises(AssertionError):
        splits.auditar_fuga(df.iloc[:3], df.iloc[3:6], df.iloc[6:], eventos)


def test_ventanas_no_cruzan_huecos(entorno):
    idx = entorno["s4"].index
    st = windows.inicios_validos(idx, 60, 10)
    d = np.diff(idx.values).astype("timedelta64[m]").astype(int)
    pref = np.r_[0, np.cumsum(d != 1)]
    assert np.all(pref[st + 69] - pref[st] == 0)
    com = windows.inicios_comunes(idx, [30, 60], 10)
    assert np.array_equal(com[30] + 29, com[60] + 59)   # mismo instante t


def test_particiones_y_persistencia(entorno):
    cfg = entorno["cfg"]
    d = preparar_datos(cfg, verbose=False)
    parts = construir_particiones(cfg, d, ft.CONJUNTOS["E_completo"])
    p = parts["val"]
    # Y real coincide con el S4 del DataFrame en t+1
    t1 = p.t[0] + pd.Timedelta(minutes=1)
    assert np.isclose(p.y_real[0, 0], d["val"].loc[t1, "S4"])
    # el escalador se ajustó solo con Train
    assert np.isclose(parts["escalador"].data_max_[0], d["train"]["S4"].max())
    yb = baselines(cfg, d, parts, "val")
    assert yb["Persistencia"].shape == p.y_real.shape
    fila = ev.evaluar(p, yb["Persistencia"], "Persistencia")
    assert 0 <= fila["POD"] <= 1


def test_contingencia_a_mano():
    evento = np.array([1, 1, 0, 0, 1, 0], bool)
    alerta = np.array([1, 0, 1, 0, 1, 0], bool)
    c = ev.contingencia(evento, alerta)
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (2, 1, 1, 2)
    assert np.isclose(c["POD"], 2 / 3) and np.isclose(c["Precision"], 2 / 3)


def test_estadisticas():
    rng = np.random.default_rng(0)
    n, H = 3000, 10
    dia = pd.date_range("2025-01-01", periods=n, freq="30min").normalize()
    y = rng.random((n, H))
    a = y + rng.normal(0, 0.3, (n, H))
    b = y + rng.normal(0, 0.05, (n, H))
    r = stats.bootstrap_bloques(y, a, b, dia, stats.m_rmse_global, n_boot=200)
    assert r["significativo"] and r["diferencia"] > 0
    dm = stats.diebold_mariano(y, a, b, dia)
    assert dm["p_valor"] < 0.01
    igual = stats.bootstrap_bloques(y, a, a, dia, stats.m_rmse_global, n_boot=50)
    assert not igual["significativo"]
