from .baselines import persistencia, climatologia_diaria, ajustar_climatologia  # noqa: F401
from .lstm import construir_lstm  # noqa: F401


def construir_modelo(cfg_modelo: dict, input_shape, horizon, perdida, learning_rate=1e-3, seed=42):
    """Fábrica única: la misma llamada sirve para LSTM y PatchTST."""
    tipo = cfg_modelo["tipo"]
    if tipo == "patchtst":
        from .patchtst import construir_patchtst
        return construir_patchtst(input_shape, horizon, perdida, learning_rate, seed=seed,
                                  **{k: v for k, v in cfg_modelo.items() if k != "tipo"})
    return construir_lstm(tipo, input_shape, horizon, perdida, learning_rate,
                          cfg_modelo.get("units1", 64), cfg_modelo.get("units2", 64),
                          cfg_modelo.get("dropout", 0.3), seed)
