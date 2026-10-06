"""PatchTST (Nie et al., ICLR 2023) en Keras, adaptado a pronóstico de S4.

Diseño fiel al original:
- Independencia de canal: cada variable se procesa por separado con el MISMO
  codificador Transformer (pesos compartidos): los canales se apilan en el eje
  del lote, así ningún token atiende a otro canal.
- Parches: la ventana de LOOKBACK se divide en parches de `patch_len` con paso
  `stride`; cada parche es un token. Con LOOKBACK=360, 16/8 -> 44 tokens.
- RevIN: normalización por instancia y canal; la salida se desnormaliza con la
  media/desviación del canal objetivo (S4, posición `idx_target`).
- Cabeza: concatena las representaciones de todos los canales y proyecta al
  horizonte de S4 (el original predice cada canal; aquí solo interesa S4).
"""
from __future__ import annotations


def construir_patchtst(input_shape, horizon, perdida, learning_rate=1e-3, seed=42,
                       patch_len=16, stride=8, d_model=64, n_heads=4, n_capas=2,
                       d_ff=128, dropout=0.2, idx_target=0, **_):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    L, C = input_shape
    P = (L - patch_len) // stride + 1

    class BloqueEncoder(layers.Layer):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.att = layers.MultiHeadAttention(n_heads, d_model // n_heads, dropout=dropout)
            self.n1, self.n2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
            self.f1, self.f2 = layers.Dense(d_ff, activation="gelu"), layers.Dense(d_model)
            self.d1, self.d2, self.d3 = layers.Dropout(dropout), layers.Dropout(dropout), layers.Dropout(dropout)

        def call(self, x, training=False):
            x = self.n1(x + self.d1(self.att(x, x, training=training), training=training))
            f = self.f2(self.d2(self.f1(x), training=training))
            return self.n2(x + self.d3(f, training=training))

    class PatchTST(keras.Model):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.emb = layers.Dense(d_model)
            self.pos = self.add_weight(name="pos", shape=(1, P, d_model), initializer="random_normal")
            self.drop = layers.Dropout(dropout)
            self.bloques = [BloqueEncoder() for _ in range(n_capas)]
            self.head_drop = layers.Dropout(dropout)
            self.head = layers.Dense(horizon)

        def call(self, x, training=False):
            x = tf.transpose(x, [0, 2, 1])                                 # (B, C, L)
            media = tf.reduce_mean(x, axis=-1, keepdims=True)
            desv = tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-5
            x = (x - media) / desv
            x = tf.signal.frame(x, patch_len, stride, axis=-1)            # (B, C, P, patch_len)
            b = tf.shape(x)[0]
            x = tf.reshape(x, [-1, P, patch_len])                         # canales en el eje del lote
            x = self.drop(self.emb(x) + self.pos, training=training)
            for blq in self.bloques:
                x = blq(x, training=training)
            x = tf.reshape(x, [b, C * P * d_model])
            y = self.head(self.head_drop(x, training=training))           # normalizado (RevIN)
            return y * desv[:, idx_target, :] + media[:, idx_target, :]  # escala del modelo

    model = PatchTST(name=f"PatchTST_LB{L}_H{horizon}_p{patch_len}s{stride}")
    model(tf.zeros((1, L, C)))          # construye todas las capas
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=perdida, metrics=["mae"])
    return model
