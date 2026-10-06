"""Función de pérdida Weighted Focal MSE (WFL) para regresión multi-paso.

FÓRMULA EXACTA IMPLEMENTADA (esta es la que debe aparecer en la tesis, el
artículo y el informe final; las tres versiones actuales de los documentos
difieren entre sí y ninguna coincide con el código):

    e_i   = ŷ_i - y_i
    w_i   = β              si y_i >= u        (u = umbral de evento en escala del modelo)
            1              en otro caso
    f_i   = (|e_i| + ε)^α                     (término focal)
    p_i   = γ_s            si y_i >= u  y  ŷ_i < ρ·y_i   (subestimación de un evento)
            1              en otro caso
    L     = mean( e_i² · w_i · f_i · p_i )

Valores vigentes: α = 1.5, β = 100, γ_s = 2.0, ρ = 0.8, ε = 1e-7.
"""
from __future__ import annotations


def weighted_focal_mse(umbral, alpha=1.5, beta=100.0, penal_subestimacion=2.0,
                       ratio_subestimacion=0.8, eps=1e-7):
    import tensorflow as tf

    def loss(y_true, y_pred):
        error = y_pred - y_true
        evento = y_true >= umbral
        w = tf.where(evento, beta, 1.0)
        focal = tf.pow(tf.abs(error) + eps, alpha)
        sub = tf.logical_and(evento, y_pred < y_true * ratio_subestimacion)
        p = tf.where(sub, penal_subestimacion, 1.0)
        return tf.reduce_mean(tf.square(error) * w * focal * p)

    loss.__name__ = f"wfl_a{alpha}_b{beta}"
    return loss


def construir_perdida(cfg_perdida: dict, umbral_escalado: float):
    if cfg_perdida.get("tipo", "wfl") == "mse":
        return "mse"
    return weighted_focal_mse(umbral_escalado, cfg_perdida["alpha"], cfg_perdida["beta"],
                              cfg_perdida.get("penal_subestimacion", 2.0),
                              cfg_perdida.get("ratio_subestimacion", 0.8))
