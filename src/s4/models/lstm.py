"""Arquitecturas LSTM (migradas de la sección 25 del notebook, sin cambios de diseño):
misma cabeza densa para las tres -> la única diferencia es la capa recurrente."""
from __future__ import annotations


def construir_lstm(tipo, input_shape, horizon, perdida, learning_rate=1e-3,
                   units1=64, units2=64, dropout=0.3, seed=42):
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    capas = [layers.Input(shape=input_shape)]
    if tipo == "simple":
        capas += [layers.LSTM(units1), layers.Dropout(dropout)]
    elif tipo == "apilado":
        capas += [layers.LSTM(units1, return_sequences=True), layers.Dropout(dropout),
                  layers.LSTM(units2), layers.Dropout(dropout)]
    elif tipo == "bidireccional":
        capas += [layers.Bidirectional(layers.LSTM(units1, return_sequences=True)),
                  layers.BatchNormalization(), layers.Dropout(dropout),
                  layers.Bidirectional(layers.LSTM(units2)),
                  layers.BatchNormalization(), layers.Dropout(dropout)]
    else:
        raise ValueError(f"tipo LSTM desconocido: {tipo}")
    capas += [layers.Dense(64, activation="relu"), layers.Dense(horizon, activation="linear")]
    model = Sequential(capas, name=f"LSTM_{tipo}_LB{input_shape[0]}_H{horizon}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=perdida, metrics=["mae"])
    return model
