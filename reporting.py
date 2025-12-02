import json

def imprimir_resumen_consola(task_name, modelo, metricas):
    print(f"\n--- RESUMEN AGREGADO DE LA TAREA: {task_name} ---")
    print(f"Modelo: {modelo}")
    print("-" * 50)
    if metricas:
        for key, value in metricas.items():
            if isinstance(value, dict):
                print(f"{key.replace('_', ' ').capitalize()}:")
                for sub_key, sub_value in value.items():
                    print(f"  - {sub_key.replace('_', ' ').capitalize()}: {sub_value}")
            else:
                print(f"{key.replace('_', ' ').capitalize()}: {value}")
    else:
        print("No se pudieron calcular métricas para esta tarea.")
    print("-" * 50)


def generar_reporte_final(config, evaluacion_por_tarea, total_prompt_tokens, total_eval_tokens):
    return {
        "modelo_evaluado": config.get("modelo_a_evaluar"),
        "directorio_imagenes": config.get("directorio_imagenes"),
        "hiperparametros": config.get("hiperparametros"),
        "estadisticas_tokens": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_eval_tokens": total_eval_tokens,
            "total_tokens": total_prompt_tokens + total_eval_tokens
        },
        "evaluacion_por_tarea": evaluacion_por_tarea
    }