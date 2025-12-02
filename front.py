import streamlit as st
from ruamel.yaml import YAML
import subprocess
import os
import sys
import locale
import re
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_FILE = "config.yaml"
SCRIPT_A_EJECUTAR = "main.py"
PALABRA_CLAVE_PROGRESO = "--- Procesando imagen:"
REPORTS_DIR = "reportes"

def modificar_config_completo(nombre_modelo, dir_imagenes, tareas_seleccionadas):
    yaml = YAML()
    yaml.preserve_quotes = True

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
    except Exception as e:
        st.error(f"Error al leer '{CONFIG_FILE}': {e}")
        return False

    config['modelo_a_evaluar'] = nombre_modelo
    config['directorio_imagenes'] = dir_imagenes
    config['tareas_a_ejecutar'] = tareas_seleccionadas

    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        return True
    except IOError as e:
        st.error(f"Error al escribir en '{CONFIG_FILE}': {e}")
        return False

def contar_imagenes(directorio):
    if not os.path.isdir(directorio): return 0
    extensiones_validas = ('.png', '.jpg', '.jpeg')
    count = 0
    for f in os.listdir(directorio):
        if f.lower().endswith(extensiones_validas):
            json_filename = os.path.splitext(f)[0] + ".json"
            if os.path.exists(os.path.join(directorio, json_filename)):
                count += 1
    return count

def ejecutar_evaluacion_st(modelo_key, directorio_imagenes, progress_bar, progress_status, log_placeholder):
    if not os.path.exists(SCRIPT_A_EJECUTAR):
        st.error(f"Error: No se encuentra el script '{SCRIPT_A_EJECUTAR}'.")
        return False

    total_imagenes = contar_imagenes(directorio_imagenes)
    progress_status.text(f"Encontradas {total_imagenes} imágenes para procesar.")
    imagenes_procesadas = 0
    full_log = "Iniciando la ejecución del script...\n"
    log_placeholder.code(full_log, language="log")

    try:
        proceso = subprocess.Popen(
            [sys.executable, "-u", SCRIPT_A_EJECUTAR], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding=locale.getpreferredencoding(), errors='replace'
        )
        for linea in iter(proceso.stdout.readline, ''):
            linea_limpia = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', linea)
            full_log += linea_limpia
            log_placeholder.code(full_log, language="log")
            if PALABRA_CLAVE_PROGRESO in linea_limpia:
                imagenes_procesadas += 1
                if total_imagenes > 0:
                    progreso = min(imagenes_procesadas / total_imagenes, 1.0)
                    progress_bar.progress(progreso)
                    progress_status.text(f"Procesando: {imagenes_procesadas} / {total_imagenes} imágenes")
        proceso.wait()
        if total_imagenes > 0:
            progress_bar.progress(1.0)
            progress_status.text(f"Procesadas: {total_imagenes} / {total_imagenes} imágenes. ¡Completado!")
        return proceso.returncode == 0
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al ejecutar el script: {e}")
        return False

def listar_subdirectorios(path="."):
    try:
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        return []

def cargar_y_procesar_resultados(directorio_reportes):
    hF1_data, hallucination_data, pds_data = [], [], []
    archivos_json = glob.glob(os.path.join(directorio_reportes, "*.json"))

    if not archivos_json:
        st.warning(f"No se encontraron archivos de resultados en '{directorio_reportes}'.")
        return None, None, None, None, None, None

    total_preguntas_hallucination = 0

    for archivo in archivos_json:
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
            modelo = data.get("modelo_evaluado", "Desconocido")
            eval_tareas = data.get("evaluacion_por_tarea", {})

            if "hF1" in eval_tareas:
                metricas = eval_tareas["hF1"].get("metricas", {})
                ganancia = metricas.get("ganancia_multimodal", {})
                metricas_multi = metricas.get("metricas_multimodal", {})
                hF1_data.append({
                    "Modelo": modelo, "Ganancia hF1 Promedio": ganancia.get("sobre_hF1_promedio"),
                    "Ganancia hF1 Global": ganancia.get("sobre_hF1_global"), "hF1 Promedio": metricas_multi.get("hF1_promedio"),
                    "hF1 Global": metricas_multi.get("hF1_global"), "Distancia LCA Errores": metricas_multi.get("distancia_LCA_promedio_en_errores")
                })
            
            if "Hallucination" in eval_tareas:
                metricas = eval_tareas["Hallucination"].get("metricas", {})
                if "total_preguntas_en_la_tarea" in metricas:
                    total_preguntas_hallucination = metricas["total_preguntas_en_la_tarea"]
                hallucination_data.append({
                    "Modelo": modelo, "Falsos Positivos": metricas.get("falsos_positivos_detectados"),
                    "Falsos Negativos": metricas.get("falsos_negativos_detectador"), "% Tasa Alucinación": metricas.get("tasa_de_alucinacion_percent")
                })

            if "PDS" in eval_tareas:
                rendimiento = eval_tareas["PDS"].get("metricas", {}).get("rendimiento_por_nivel", {})
                for nivel, valores in rendimiento.items():
                    pds_data.append({
                        "Modelo": modelo, "Nivel Distorsión": nivel.replace("Nivel_", ""),
                        "hF1 Global Nivel": valores.get("hF1_global_nivel"), 
                        "% Caída Rendimiento": valores.get("caida_rendimiento_percent")
                    })
        except Exception as e:
            st.error(f"Error al procesar el archivo '{archivo}': {e}")
    
    df_hf1 = pd.DataFrame(hF1_data)
    df_hall = pd.DataFrame(hallucination_data)
    df_pds = pd.DataFrame(pds_data)

    def calcular_score(df, metricas_config):
        if df.empty:
            return df
        
        scores = pd.DataFrame(index=df.index)
        num_metricas = 0

        for col, config in metricas_config.items():
            if col in df.columns:
                num_metricas += 1
                min_val, max_val = config['min'], config['max']
                if max_val > min_val:
                    normalized = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                
                if config.get('invertir', False):
                    scores[col] = 1 - normalized
                else:
                    scores[col] = normalized
        
        if num_metricas > 0:
            ponderacion = 100 / num_metricas
            df['Score'] = scores.sum(axis=1) * ponderacion
        else:
            df['Score'] = 0

        cols = ['Score'] + [col for col in df.columns if col != 'Score']
        return df[cols]

    metricas_hf1 = {
        "Ganancia hF1 Promedio": {'min': 0, 'max': 1}, "Ganancia hF1 Global": {'min': 0, 'max': 1},
        "hF1 Promedio": {'min': 0, 'max': 1}, "hF1 Global": {'min': 0, 'max': 1},
        "Distancia LCA Errores": {'min': 0, 'max': 6, 'invertir': True}
    }
    df_hf1 = calcular_score(df_hf1, metricas_hf1)

    metricas_hall = {
        "Falsos Positivos": {'min': 0, 'max': total_preguntas_hallucination, 'invertir': True},
        "Falsos Negativos": {'min': 0, 'max': total_preguntas_hallucination, 'invertir': True},
        "% Tasa Alucinación": {'min': 0, 'max': 100, 'invertir': True}
    }
    df_hall = calcular_score(df_hall, metricas_hall)
    
    metricas_pds_agg = {
        "PDS_hF1_Promedio": {'min': 0, 'max': 1},
        "PDS_Caida_Promedio_pc": {'min': 0, 'max': 100, 'invertir': True},
        "PDS_Caida_Maxima_pc": {'min': 0, 'max': 100, 'invertir': True}
    }
    
    if not df_pds.empty:
        df_pds_agg = df_pds.groupby('Modelo').agg(
            PDS_hF1_Promedio=('hF1 Global Nivel', 'mean'),
            PDS_Caida_Promedio_pc=('% Caída Rendimiento', 'mean'),
            PDS_Caida_Maxima_pc=('% Caída Rendimiento', 'max')
        ).reset_index()
        df_pds_agg = calcular_score(df_pds_agg, metricas_pds_agg)
    else:
        df_pds_agg = pd.DataFrame(columns=['Modelo', 'Score'])

    df_global = pd.DataFrame()
    if not df_hf1.empty:
        df_global = df_hf1
        if not df_hall.empty:
            df_global = pd.merge(df_global, df_hall, on="Modelo", how="outer")
        if not df_pds_agg.empty:
            df_global = pd.merge(df_global, df_pds_agg.rename(columns={'Score': 'Score_pds'}), on="Modelo", how="outer")
    elif not df_hall.empty:
        df_global = df_hall
        if not df_pds_agg.empty:
            df_global = pd.merge(df_global, df_pds_agg.rename(columns={'Score': 'Score_pds'}), on="Modelo", how="outer")
    else:
        df_global = df_pds_agg

    metricas_global = {**metricas_hf1, **metricas_hall, **metricas_pds_agg}
    df_global = calcular_score(df_global, metricas_global)
    df_global = df_global.drop(columns=['Score_x', 'Score_y', 'Score_pds'], errors='ignore')

    return hF1_data, hallucination_data, pds_data, df_global, df_pds_agg, df_pds

def highlight_top_values(s, ascending=False):
    if not pd.api.types.is_numeric_dtype(s):
        return ['' for _ in s]

    ranks = s.rank(method='min', ascending=ascending)
    
    gold = 'background-color: gold; color: black;'
    silver = 'background-color: silver; color: black;'
    bronze = 'background-color: #CD7F32; color: white;'  
    default = ''
    
    styles = []
    for rank in ranks:
        if rank == 1.0:
            styles.append(gold)
        elif rank == 2.0:
            styles.append(silver)
        elif rank == 3.0:
            styles.append(bronze)
        else:
            styles.append(default)
    return styles

def mostrar_dashboard_resultados(hF1_data, hallucination_data, pds_data, df_global, df_pds_agg, df_pds_detallado):
    st.header("Dashboard de Resultados Comparativos")

    tab_global, tab_hf1, tab_hall, tab_pds = st.tabs([
        "🏆 Clasificación Global", "Clasificación hF1", "Clasificación Hallucination", 
        "Clasificación PDS"
    ])

    metricas_positivas = [
        "Score", "Ganancia hF1 Promedio", "Ganancia hF1 Global", "hF1 Promedio", 
        "hF1 Global", "PDS_hF1_Promedio", "hF1 Global Nivel"
    ]
    metricas_negativas = [
        "Distancia LCA Errores", "Falsos Positivos", "Falsos Negativos", 
        "% Tasa Alucinación", "PDS_Caida_Promedio_pc", "PDS_Caida_Maxima_pc", 
        "% Caída Rendimiento"
    ]

    def _plot_static_score_bar(series, title=None):
        if series.empty:
            st.info("Serie vacía: no hay datos para graficar.")
            return
        
        series = series.sort_values(ascending=False)
        
        fig, ax = plt.subplots()
        indices = range(len(series))
        bars = ax.bar(indices, series.values)
        ax.set_ylim(0, 100)
        ax.set_xticks(indices)
        ax.set_xticklabels([str(x) for x in series.index], rotation=45, ha='right')
        ax.set_ylabel('Score')
        if title:
            ax.set_title(title, pad=20)

        for rect, val in zip(bars, series.values):
            height = rect.get_height()
            ax.annotate(f"{val:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_global:
        if not df_global.empty:
            df_global = df_global.set_index("Modelo")
            
            positive_cols_in_df = [col for col in metricas_positivas if col in df_global.columns]
            negative_cols_in_df = [col for col in metricas_negativas if col in df_global.columns]
            
            st.dataframe(df_global.style
                .apply(highlight_top_values, ascending=False, subset=positive_cols_in_df)
                .apply(highlight_top_values, ascending=True, subset=negative_cols_in_df)
                .format("{:.4f}").highlight_null('lightgray')
            )

            try:
                if 'Score' in df_global.columns:
                    st.write("### Gráfica: Score por modelo (Global)")
                    _plot_static_score_bar(df_global['Score'], title="Score por modelo (Global)")
            except Exception as e:
                st.error(f"Error al crear la gráfica global: {e}")
        else:
            st.info("No hay datos globales para mostrar.")

    with tab_hf1:
        if hF1_data:
            df = pd.DataFrame(hF1_data)
            metricas_hf1 = {
                "Ganancia hF1 Promedio": {'min': 0, 'max': 1}, "Ganancia hF1 Global": {'min': 0, 'max': 1},
                "hF1 Promedio": {'min': 0, 'max': 1}, "hF1 Global": {'min': 0, 'max': 1},
                "Distancia LCA Errores": {'min': 0, 'max': 6, 'invertir': True}
            }
            def calcular_score(d, m):
                if d.empty: return d
                sc = pd.DataFrame(index=d.index); n_m = 0
                for c, cfg in m.items():
                    if c in d.columns:
                        n_m += 1; min_v, max_v = cfg['min'], cfg['max']
                        norm = (d[c] - min_v) / (max_v - min_v) if max_v > min_v else 0.5
                        sc[c] = 1 - norm if cfg.get('invertir', False) else norm
                d['Score'] = (sc.sum(axis=1) * (100 / n_m)) if n_m > 0 else 0
                return d[['Score'] + [c for c in d.columns if c != 'Score']]
            df = calcular_score(df, metricas_hf1).set_index("Modelo")

            positive_cols_in_df = [col for col in ["Score", "Ganancia hF1 Promedio", "Ganancia hF1 Global", "hF1 Promedio", "hF1 Global"] if col in df.columns]
            negative_cols_in_df = [col for col in ["Distancia LCA Errores"] if col in df.columns]

            st.dataframe(df.style
                .apply(highlight_top_values, ascending=False, subset=positive_cols_in_df)
                .apply(highlight_top_values, ascending=True, subset=negative_cols_in_df)
                .format("{:.4f}")
            )
            try:
                if 'Score' in df.columns:
                    st.write("### Gráfica: Score por modelo (hF1)")
                    _plot_static_score_bar(df['Score'], title="Score por modelo (hF1)")
            except Exception as e:
                st.error(f"Error al crear la gráfica hF1: {e}")
        else:
            st.info("No hay datos de hF1 para mostrar.")

    with tab_hall:
        if hallucination_data:
            df = pd.DataFrame(hallucination_data)
            total_preguntas_hallucination = 0
            archivos_json = glob.glob(os.path.join(REPORTS_DIR, "*.json"))
            for archivo in archivos_json:
                with open(archivo, 'r', encoding='utf-8') as f: data = json.load(f)
                if "Hallucination" in data.get("evaluacion_por_tarea", {}):
                    total_preguntas_hallucination = data["evaluacion_por_tarea"]["Hallucination"].get("metricas", {}).get("total_preguntas_en_la_tarea", 0)
                    if total_preguntas_hallucination > 0: break
            
            metricas_hall = {
                "Falsos Positivos": {'min': 0, 'max': total_preguntas_hallucination, 'invertir': True},
                "Falsos Negativos": {'min': 0, 'max': total_preguntas_hallucination, 'invertir': True},
                "% Tasa Alucinación": {'min': 0, 'max': 100, 'invertir': True}
            }
            def calcular_score(d, m):
                if d.empty: return d
                sc = pd.DataFrame(index=d.index); n_m = 0
                for c, cfg in m.items():
                    if c in d.columns:
                        n_m += 1; min_v, max_v = cfg['min'], cfg['max']
                        norm = (d[c] - min_v) / (max_v - min_v) if max_v > min_v else 0.5
                        sc[c] = 1 - norm if cfg.get('invertir', False) else norm
                d['Score'] = (sc.sum(axis=1) * (100 / n_m)) if n_m > 0 else 0
                return d[['Score'] + [c for c in d.columns if c != 'Score']]
            df = calcular_score(df, metricas_hall).set_index("Modelo")
            
            positive_cols_in_df = [col for col in ["Score"] if col in df.columns]
            negative_cols_in_df = [col for col in ["Falsos Positivos", "Falsos Negativos", "% Tasa Alucinación"] if col in df.columns]

            st.dataframe(df.style
                .apply(highlight_top_values, ascending=False, subset=positive_cols_in_df)
                .apply(highlight_top_values, ascending=True, subset=negative_cols_in_df)
                .format("{:.2f}", subset=["% Tasa Alucinación", "Score"])
                .format("{:.0f}", subset=["Falsos Positivos", "Falsos Negativos"])
            )
            try:
                if 'Score' in df.columns:
                    st.write("### Gráfica: Score por modelo (Hallucination)")
                    _plot_static_score_bar(df['Score'], title="Score por modelo (Hallucination)")
            except Exception as e:
                st.error(f"Error al crear la gráfica Hallucination: {e}")
        else:
            st.info("No hay datos de Hallucination para mostrar.")

    with tab_pds:
        if not df_pds_detallado.empty:
            st.subheader("Resultados Detallados por Nivel de Distorsión")
            df_detallado_styled = df_pds_detallado.set_index(["Modelo", "Nivel Distorsión"])
            
            # <-- MODIFICADO: Se eliminan las llamadas a .apply(highlight_top_values, ...) para quitar el fondo de color.
            st.dataframe(df_detallado_styled.style.format("{:.4f}"))
            
            st.divider()

            st.subheader("Resumen y Score Agregado por Modelo")
            if not df_pds_agg.empty:
                df_agg_styled = df_pds_agg.set_index("Modelo")
                positive_cols_in_df = [col for col in ["Score", "PDS_hF1_Promedio"] if col in df_agg_styled.columns]
                negative_cols_in_df = [col for col in ["PDS_Caida_Promedio_pc", "PDS_Caida_Maxima_pc"] if col in df_agg_styled.columns]

                st.dataframe(df_agg_styled.style
                    .apply(highlight_top_values, ascending=False, subset=positive_cols_in_df)
                    .apply(highlight_top_values, ascending=True, subset=negative_cols_in_df)
                    .format("{:.4f}")
                )

                try:
                    if 'Score' in df_agg_styled.columns:
                        st.write("### Gráfica: Score por modelo (PDS)")
                        _plot_static_score_bar(df_agg_styled['Score'], title="Score por modelo (PDS)")
                except Exception as e:
                    st.error(f"Error al crear la gráfica PDS: {e}")
            else:
                st.info("No hay datos agregados de PDS para mostrar.")
        else:
            st.info("No hay datos de PDS para mostrar.")


st.set_page_config(layout="wide")
st.title("Evaluación de Modelos por Lotes")

try:
    yaml = YAML()
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config_inicial = yaml.load(f)
    modelos_disponibles = config_inicial.get("modelos_a_evaluar_en_lote", [])
    tareas_disponibles = ["hF1", "Hallucination", "PDS"]
    directorios_locales = ["."] + listar_subdirectorios()
except Exception as e:
    st.error(f"No se pudo cargar '{CONFIG_FILE}': {e}")
    st.stop()

with st.sidebar:
    st.header("Configuración de la Evaluación")
    directorio_imagenes_input = st.selectbox("Selecciona el Directorio de Imágenes", options=directorios_locales, index=0, help="Elige una carpeta disponible.")
    tareas_a_ejecutar_input = st.multiselect("Tareas a Ejecutar", options=tareas_disponibles, default=config_inicial.get("tareas_a_ejecutar", []), help="Selecciona una o más tareas.")
    st.warning("Asegúrate de que 'Ollama' esté en ejecución.")

if not modelos_disponibles:
    st.error("La clave 'modelos_a_evaluar_en_lote' no se encuentra en el config.")
else:
    st.info(f"Se evaluarán **{len(modelos_disponibles)}** modelos: **{', '.join(modelos_disponibles)}**")

    if st.button("Iniciar Evaluación", type="primary", use_container_width=True):
        if not os.path.isdir(directorio_imagenes_input):
            st.error(f"El directorio '{directorio_imagenes_input}' no existe.")
        elif not tareas_a_ejecutar_input:
            st.error("Debes seleccionar al menos una tarea.")
        else:
            st.success("Iniciando el proceso de evaluación...")
            evaluaciones_exitosas = True
            
            for modelo in modelos_disponibles:
                st.divider()
                st.header(f"Procesando Modelo: {modelo}")
                if not modificar_config_completo(modelo, directorio_imagenes_input, tareas_a_ejecutar_input):
                    st.error(f"No se pudo modificar config para '{modelo}'. Saltando.")
                    continue
                
                progress_status = st.empty()
                progress_bar = st.progress(0)

                with st.expander(f"Ver log de la consola para '{modelo}'"):
                    log_placeholder = st.empty()

                exito = ejecutar_evaluacion_st(modelo, directorio_imagenes_input, progress_bar, progress_status, log_placeholder)
                if exito:
                    st.success(f"Evaluación para '{modelo}' finalizada.")
                else:
                    st.error(f"La evaluación para '{modelo}' falló.")
                    evaluaciones_exitosas = False
            
            st.divider()
            if evaluaciones_exitosas:
                st.header("¡Todas las evaluaciones han finalizado!")
            else:
                st.warning("Algunas evaluaciones finalizaron con errores.")
            
            hF1, hall, pds, df_global, df_pds_agg, df_pds_detallado = cargar_y_procesar_resultados(REPORTS_DIR)
            
            if hF1 or hall or pds:
                mostrar_dashboard_resultados(hF1, hall, pds, df_global, df_pds_agg, df_pds_detallado)