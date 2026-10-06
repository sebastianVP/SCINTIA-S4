"""
Sistema de Pronóstico de Centelleo Ionosférico (S4) con LSTM
==============================================================

Modo "explorador": subes un tramo histórico (cuantas más filas, mejor --
no solo 360), y dentro de la app te mueves libremente por el tiempo para
ver el pronóstico del modelo superpuesto contra el valor REAL que
efectivamente ocurrió (backtesting visual). Útil para validar el modelo
en distintos regímenes: eventos fuertes vs. períodos tranquilos.

Si subes un archivo con EXACTAMENTE LOOKBACK filas (el caso de producción
real, sin futuro conocido), la app cae automáticamente en modo pronóstico
puro -- sin comparación, porque no hay con qué comparar.

Requiere, en la misma carpeta (o ajusta ASSETS_DIR abajo):
    - modelo_S4_lstm.keras
    - SCALER_S4_app.pkl
    - config.json
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
st.set_page_config(page_title="Pronóstico S4 Ionosférico", layout="wide")

ASSETS_DIR = Path(__file__).parent
MODEL_PATH = ASSETS_DIR / "modelo_S4_lstm.keras"
SCALER_PATH = ASSETS_DIR / "SCALER_S4_app.pkl"
CONFIG_PATH = ASSETS_DIR / "config.json"


# =============================================================================
# 2. CARGA EN CACHÉ DE RECURSOS
# =============================================================================
@st.cache_resource
def load_assets():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    scaler = joblib.load(SCALER_PATH)
    # Solo inferencia: no hace falta el optimizador ni el loss personalizado.
    model = load_model(MODEL_PATH, compile=False)

    features_cols = config["FEATURES_COLS"]
    target_col = config["TARGET_COL"]
    idx_target = features_cols.index(target_col)
    min_target = scaler.data_min_[idx_target]
    max_target = scaler.data_max_[idx_target]

    return model, scaler, config, min_target, max_target


# =============================================================================
# 3. FEATURE ENGINEERING (idéntico al del entrenamiento -- verificado)
# =============================================================================
def add_temporal_features(df):
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    minute_of_day = idx.hour * 60 + idx.minute
    out["daily_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["daily_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    days_in_year = np.where(idx.is_leap_year, 366, 365)
    out["day_of_year_sin"] = np.sin(2 * np.pi * (idx.dayofyear - 1) / days_in_year)
    out["day_of_year_cos"] = np.cos(2 * np.pi * (idx.dayofyear - 1) / days_in_year)
    return out


def desnormalizar(values_scaled, min_v, max_v):
    return np.abs(values_scaled * (max_v - min_v) + min_v)


def verificar_continuidad(index, sampling_minutes):
    if len(index) < 2:
        return 0
    diffs = np.diff(index.values).astype("timedelta64[m]").astype(int)
    return int(np.sum(diffs != sampling_minutes))


# =============================================================================
# 4. LÓGICA DE EXPLORACIÓN (posiciones de corte, búsqueda de evento/calma)
# =============================================================================
def rango_posiciones_validas(n_filas, lookback, horizon):
    """pos_min: primera posición con LOOKBACK filas completas antes.
    pos_max_con_comparacion: última posición con HORIZON filas reales
    completas después (para poder comparar). Puede no haber ninguna."""
    pos_min = lookback - 1
    pos_max_sin_comparacion = n_filas - 1
    pos_max_con_comparacion = n_filas - horizon - 1
    return pos_min, pos_max_sin_comparacion, pos_max_con_comparacion


def future_max_por_posicion(target_array, horizon):
    """future_max[pos] = max(target[pos+1 : pos+1+horizon]). Vectorizado."""
    n = len(target_array)
    if n <= horizon:
        return np.array([])
    windows = np.lib.stride_tricks.sliding_window_view(target_array[1:], horizon)
    return windows.max(axis=1)


def buscar_evento_mas_fuerte(target_array, pos_min, pos_max_con_comparacion, horizon):
    future_max = future_max_por_posicion(target_array, horizon)
    if pos_max_con_comparacion < pos_min or len(future_max) <= pos_max_con_comparacion:
        return None
    tramo = future_max[pos_min : pos_max_con_comparacion + 1]
    if len(tramo) == 0:
        return None
    return pos_min + int(np.argmax(tramo))


def buscar_periodo_calmo(target_array, pos_min, pos_max_con_comparacion, horizon):
    future_max = future_max_por_posicion(target_array, horizon)
    if pos_max_con_comparacion < pos_min or len(future_max) <= pos_max_con_comparacion:
        return None
    tramo = future_max[pos_min : pos_max_con_comparacion + 1]
    if len(tramo) == 0:
        return None
    return pos_min + int(np.argmin(tramo))


def listar_eventos_detectados(target_array, index, threshold, horizon, pos_min,
                               pos_max_con_comparacion, max_eventos=150):
    """Detecta segmentos contiguos de evento (S4 >= threshold) y devuelve,
    por cada uno, la posición de corte justo ANTES de que empiece (para que
    el horizonte de predicción tenga que cubrirlo), su timestamp y su pico
    real. Si hay demasiados eventos para un desplegable usable, se queda
    con los `max_eventos` de mayor pico (y los reordena cronológicamente)."""
    es_evento = target_array >= threshold
    cambia = np.diff(es_evento.astype(int), prepend=0)
    inicios = np.where(cambia == 1)[0]
    fines = np.where(cambia == -1)[0] - 1
    if len(fines) < len(inicios):
        fines = np.append(fines, len(target_array) - 1)

    eventos = []
    for ini, fin in zip(inicios, fines):
        pos_corte_evento = ini - 1
        if pos_corte_evento < pos_min or pos_corte_evento > pos_max_con_comparacion:
            continue  # evento sin suficiente historia antes, o sin horizonte completo después
        pico = float(target_array[ini : fin + 1].max())
        eventos.append({"pos": int(pos_corte_evento), "inicio": index[ini], "pico": pico})

    eventos.sort(key=lambda e: e["inicio"])
    if len(eventos) > max_eventos:
        eventos = sorted(eventos, key=lambda e: -e["pico"])[:max_eventos]
        eventos.sort(key=lambda e: e["inicio"])
    return eventos


def graficar_overview(index, target_array, threshold, pos_corte, max_puntos=3000):
    """Mini-gráfica de TODA la serie (downsampled tomando el máximo por
    bucket, para no perder eventos angostos al comprimir), con los eventos
    marcados en rojo y una línea vertical en el punto de corte actual --
    da contexto visual antes de mover el slider a ciegas."""
    n = len(target_array)
    bucket = max(1, n // max_puntos)
    n_buckets = int(np.ceil(n / bucket))
    resto = n_buckets * bucket - n
    padded = np.concatenate([target_array, np.full(resto, np.nan)]) if resto > 0 else target_array
    matriz = padded.reshape(n_buckets, bucket)
    valores_bucket = np.nanmax(matriz, axis=1)
    pos_bucket = np.clip(np.arange(n_buckets) * bucket + bucket // 2, 0, n - 1)
    tiempos_bucket = index[pos_bucket]

    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.plot(tiempos_bucket, valores_bucket, color="#999999", linewidth=0.8)
    es_evt = valores_bucket >= threshold
    ax.scatter(np.array(tiempos_bucket)[es_evt], valores_bucket[es_evt], color="#FF4B4B", s=10, zorder=3)
    ax.axhline(threshold, color="#FFA500", linestyle=":", linewidth=1)
    ax.axvline(index[pos_corte], color="#3498DB", linewidth=1.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return fig


# =============================================================================
# 5. PREDICCIÓN Y ARMADO DEL GRÁFICO COMPARATIVO
# =============================================================================
def predecir_en_posicion(df_fe, df_crudo, pos_corte, model, scaler, features_cols, target_col,
                          lookback, horizon, min_target, max_target):
    ventana = df_fe.iloc[pos_corte - lookback + 1 : pos_corte + 1][features_cols]
    ventana_scaled = scaler.transform(ventana)
    X_input = ventana_scaled.reshape(1, lookback, len(features_cols))
    y_pred_scaled = model.predict(X_input, verbose=0)[0]
    y_pred_real = desnormalizar(y_pred_scaled, min_target, max_target)

    cutoff_time = df_crudo.index[pos_corte]
    idx_futuro = df_crudo.index[pos_corte + 1 : pos_corte + 1 + horizon]
    return y_pred_real, cutoff_time, idx_futuro


def construir_grafico_comparativo(df_crudo, target_col, pos_corte, minutos_contexto,
                                   y_pred_real, cutoff_time, idx_futuro, horizon):
    idx_historia = df_crudo.index[max(0, pos_corte - minutos_contexto + 1) : pos_corte + 1]
    serie_obs = df_crudo[target_col].reindex(idx_historia)

    chart_df = pd.DataFrame({"S4 observado": serie_obs})

    hay_comparacion = len(idx_futuro) == horizon
    serie_pred = pd.Series(y_pred_real[: len(idx_futuro)], index=idx_futuro)
    conexion = pd.Series([serie_obs.iloc[-1]], index=[cutoff_time])
    serie_pred_c = pd.concat([conexion, serie_pred])
    chart_df = chart_df.reindex(chart_df.index.union(serie_pred_c.index))
    chart_df["S4 pronosticado"] = serie_pred_c.reindex(chart_df.index)

    rmse_ventana = None
    if hay_comparacion:
        serie_real = df_crudo[target_col].reindex(idx_futuro)
        serie_real_c = pd.concat([conexion, serie_real])
        chart_df = chart_df.reindex(chart_df.index.union(serie_real_c.index))
        chart_df["S4 real"] = serie_real_c.reindex(chart_df.index)
        rmse_ventana = float(np.sqrt(np.mean((serie_real.to_numpy() - serie_pred.to_numpy()) ** 2)))

    return chart_df.sort_index(), hay_comparacion, rmse_ventana


# =============================================================================
# 6. INTERFAZ
# =============================================================================
def main():
    st.title("🛰️ Pronóstico de Centelleo Ionosférico (S4)")

    try:
        model, scaler, config, min_target, max_target = load_assets()
    except Exception as e:
        st.error(
            f"❌ No se pudieron cargar los archivos del modelo ({e}). Verifica que "
            f"'modelo_S4_lstm.keras', 'SCALER_S4_app.pkl' y 'config.json' estén junto a esta app."
        )
        st.stop()

    LOOKBACK = config["LOOKBACK"]
    HORIZON = config["HORIZON"]
    THRESHOLD = config["THRESHOLD"]
    FEATURES_COLS = config["FEATURES_COLS"]
    TARGET_COL = config["TARGET_COL"]
    SAMPLING_MINUTES = config["SAMPLING_MINUTES"]

    with st.sidebar:
        st.success("✅ Modelo y scaler cargados.")
        st.caption(
            f"Arquitectura: LSTM {config.get('ARQUITECTURA', '?')} "
            f"({config.get('UNITS1', '?')}/{config.get('UNITS2', '?')} unidades)"
        )
        st.caption(f"LOOKBACK={LOOKBACK} min · HORIZON={HORIZON} min")
        st.caption(f"RMSE_Evento (test): {config.get('RMSE_EVENTO_TEST', float('nan')):.4f}")

    st.markdown(
        f"Sube un CSV con columnas **`Tiempo`** y **`{TARGET_COL}`**. Con al menos "
        f"**{LOOKBACK} filas** funciona en modo pronóstico puro. Si subes **más filas** "
        f"(por ejemplo, un tramo largo de tu conjunto de test), la app habilita el modo "
        f"explorador: podrás moverte por el tiempo y comparar el pronóstico contra el "
        f"valor real que efectivamente ocurrió."
    )

    uploaded_file = st.file_uploader("Archivo CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)
    time_col = "Tiempo" if "Tiempo" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if time_col is None or TARGET_COL not in df.columns:
        st.error(f"⚠️ El CSV debe tener una columna de fecha ('Tiempo') y la columna '{TARGET_COL}'.")
        st.stop()

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    if len(df) < LOOKBACK:
        st.error(f"⚠️ Se necesitan al menos {LOOKBACK} filas. El archivo tiene {len(df)}.")
        st.stop()

    df_fe = add_temporal_features(df)
    target_array = df[TARGET_COL].to_numpy()
    n_filas = len(df)

    pos_min, pos_max_sin_comp, pos_max_con_comp = rango_posiciones_validas(n_filas, LOOKBACK, HORIZON)
    modo_exploracion = pos_max_con_comp >= pos_min  # hay al menos un punto con futuro real conocido

    st.subheader(f"Archivo cargado: {n_filas:,} filas ({df.index.min()} → {df.index.max()})")

    # --- Selección del punto de corte ---
    key_slider = "pos_corte"
    if key_slider not in st.session_state:
        st.session_state[key_slider] = pos_max_sin_comp  # por defecto: el final del archivo

    if modo_exploracion:
        st.markdown("### 🔍 Modo explorador -- elige dónde \"cortar\" la historia")

        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("🔴 Ir al evento más fuerte disponible"):
            pos = buscar_evento_mas_fuerte(target_array, pos_min, pos_max_con_comp, HORIZON)
            if pos is not None:
                st.session_state[key_slider] = pos
        if col_b2.button("🟢 Ir a un período tranquilo"):
            pos = buscar_periodo_calmo(target_array, pos_min, pos_max_con_comp, HORIZON)
            if pos is not None:
                st.session_state[key_slider] = pos
        if col_b3.button("⏭️ Ir al final del archivo (sin comparación)"):
            st.session_state[key_slider] = pos_max_sin_comp

        # Desplegable con TODOS los eventos detectados (no solo el más
        # fuerte) -- cada uno salta al punto justo antes de que empiece.
        eventos = listar_eventos_detectados(target_array, df.index, THRESHOLD, HORIZON, pos_min, pos_max_con_comp)
        if eventos:
            opciones = ["-- elegir un evento específico --"] + [
                f"{e['inicio']:%Y-%m-%d %H:%M} · pico S4={e['pico']:.3f}" for e in eventos
            ]
            pos_por_opcion = {opciones[i + 1]: e["pos"] for i, e in enumerate(eventos)}

            def _saltar_a_evento_elegido():
                elegido = st.session_state["selector_evento"]
                if elegido in pos_por_opcion:
                    st.session_state[key_slider] = pos_por_opcion[elegido]

            st.selectbox(
                f"O elige entre los {len(eventos)} eventos detectados en el archivo:",
                options=opciones,
                key="selector_evento",
                on_change=_saltar_a_evento_elegido,
            )
        else:
            st.caption("No se detectaron eventos (S4 ≥ umbral) con suficiente contexto en este archivo.")

        # Mini-resumen visual: toda la serie, eventos en rojo, línea azul en
        # el punto de corte actual -- para no mover el slider a ciegas.
        st.pyplot(graficar_overview(df.index, target_array, THRESHOLD, st.session_state[key_slider]))

        pos_corte = st.slider(
            "Punto de corte (fin del historial usado por el modelo)",
            min_value=pos_min,
            max_value=pos_max_sin_comp,
            key=key_slider,
        )
        st.caption(f"Corte en: **{df.index[pos_corte]}**")
    else:
        pos_corte = pos_max_sin_comp
        st.info(
            "El archivo no tiene suficientes filas después del final para comparar contra "
            f"un valor real (se necesitarían {HORIZON} filas más) -- modo pronóstico puro."
        )

    minutos_contexto = st.slider(
        "Minutos de historia a mostrar en el gráfico",
        min_value=30,
        max_value=LOOKBACK,
        value=min(120, LOOKBACK),
        step=10,
        help=(
            f"Solo afecta lo que se GRAFICA. El modelo siempre usa los {LOOKBACK} "
            "minutos completos para predecir."
        ),
    )

    ventana_cruda = df.iloc[pos_corte - LOOKBACK + 1 : pos_corte + 1]
    n_huecos = verificar_continuidad(ventana_cruda.index, SAMPLING_MINUTES)
    if n_huecos > 0:
        st.warning(
            f"⚠️ {n_huecos} discontinuidad(es) en los {LOOKBACK} min usados como historia -- "
            "la predicción puede no ser confiable."
        )

    if st.button(f"Generar pronóstico a {HORIZON} minutos", type="primary"):
        with st.spinner("Calculando..."):
            y_pred_real, cutoff_time, idx_futuro = predecir_en_posicion(
                df_fe, df, pos_corte, model, scaler, FEATURES_COLS, TARGET_COL,
                LOOKBACK, HORIZON, min_target, max_target,
            )
            chart_df, hay_comparacion, rmse_ventana = construir_grafico_comparativo(
                df, TARGET_COL, pos_corte, minutos_contexto, y_pred_real, cutoff_time, idx_futuro, HORIZON
            )

        st.subheader(f"Observado + pronóstico desde {cutoff_time:%Y-%m-%d %H:%M}")
        colores = ["#2ECC71", "#FF4B4B", "#3498DB"] if hay_comparacion else ["#2ECC71", "#FF4B4B"]
        st.line_chart(chart_df, color=colores)

        max_pred = float(np.max(y_pred_real))
        alerta_predicha = max_pred >= THRESHOLD

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Máximo pronosticado", f"{max_pred:.3f}")
        if hay_comparacion:
            max_real = float(df[TARGET_COL].reindex(idx_futuro).max())
            alerta_real = max_real >= THRESHOLD
            col_m2.metric("Máximo real", f"{max_real:.3f}")
            col_m3.metric("RMSE de la ventana", f"{rmse_ventana:.4f}")

            if alerta_predicha == alerta_real:
                st.success(
                    f"✅ Coincidencia: el modelo {'sí' if alerta_predicha else 'no'} predijo evento, "
                    f"y {'sí' if alerta_real else 'no'} ocurrió realmente."
                )
            elif alerta_predicha and not alerta_real:
                st.warning("⚠️ Falso positivo: el modelo predijo evento, pero no ocurrió.")
            else:
                st.error("❌ Falso negativo: ocurrió un evento real que el modelo no predijo.")
        else:
            col_m2.metric("Máximo real", "no disponible")
            if alerta_predicha:
                st.warning(f"⚠️ **ALERTA DE EVENTO:** pronóstico máximo {max_pred:.4f} ≥ umbral {THRESHOLD}.")
            else:
                st.info(f"✅ Condiciones normales pronosticadas (máximo {max_pred:.4f}, umbral {THRESHOLD}).")

        tabla_pred = pd.DataFrame({"Timestamp": idx_futuro[: len(y_pred_real)], "S4 pronosticado": y_pred_real})
        if hay_comparacion:
            tabla_pred["S4 real"] = df[TARGET_COL].reindex(idx_futuro).to_numpy()
            tabla_pred["Error absoluto"] = (tabla_pred["S4 pronosticado"] - tabla_pred["S4 real"]).abs()
        st.subheader("Detalle numérico")
        st.dataframe(tabla_pred.set_index("Timestamp"))


if __name__ == "__main__":
    main()