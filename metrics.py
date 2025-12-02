#import networkx as nx
#import numpy as np
#from collections import Counter
#from scipy.optimize import linear_sum_assignment

def calcular_metricas_alucinacion(resultados_detallados):

    total_preguntas_en_la_tarea = len(resultados_detallados)
    preguntas_si_no = [res for res in resultados_detallados if isinstance(res.get('ground_truth'), str) and res['ground_truth'].lower() in ('yes', 'no')]
 
    # Contadores básicos
    TP_yes = 0  # GT yes, pred yes
    FN_yes = 0  # GT yes, pred no
    TN_no = 0   # GT no, pred no
    FP_no = 0   # GT no, pred yes (alucinaciones)
 
    for res in preguntas_si_no:
        gt = res['ground_truth'].lower()
        pred = str(res.get('respuesta_normalizada', '')).lower()
        if gt == 'yes':
            if pred == 'yes':
                TP_yes += 1
            elif pred == 'no':
                FN_yes += 1
        elif gt == 'no':
            if pred == 'yes':
                FP_no += 1
            elif pred == 'no':
                TN_no += 1
 
    negativos_reales = FP_no + TN_no  # GT == 'no'
    positivos_reales = TP_yes + FN_yes  # GT == 'yes'
 
    tasa_alucinacion = (FP_no / negativos_reales * 100) if negativos_reales > 0 else 0.0
 
    # Precision/Recall por clase 'yes'
    precision_yes = TP_yes / (TP_yes + FP_no) if (TP_yes + FP_no) > 0 else 0.0
    recall_yes = TP_yes / (TP_yes + FN_yes) if (TP_yes + FN_yes) > 0 else 0.0
    f1_yes = (2 * precision_yes * recall_yes / (precision_yes + recall_yes)) if (precision_yes + recall_yes) > 0 else 0.0
 
    # Precision/Recall por clase 'no'
    precision_no = TN_no / (TN_no + FN_yes) if (TN_no + FN_yes) > 0 else 0.0
    recall_no = TN_no / (TN_no + FP_no) if (TN_no + FP_no) > 0 else 0.0
    f1_no = (2 * precision_no * recall_no / (precision_no + recall_no)) if (precision_no + recall_no) > 0 else 0.0
 
    # Macro promedios (yes/no)
    macro_precision = (precision_yes + precision_no) / 2 if (precision_yes + precision_no) > 0 else 0.0
    macro_recall = (recall_yes + recall_no) / 2 if (recall_yes + recall_no) > 0 else 0.0
    macro_f1 = (f1_yes + f1_no) / 2 if (f1_yes + f1_no) > 0 else 0.0
 
    return {
        "metrica_principal": "Tasa de Alucinación (sobre preguntas Sí/No)",
        "total_preguntas_en_la_tarea": total_preguntas_en_la_tarea,
        "total_preguntas_si_no_relevantes": len(preguntas_si_no),
        "conteos_clase": {
            "positivos_reales_yes": positivos_reales,
            "negativos_reales_no": negativos_reales,
            "TP_yes": TP_yes,
            "FN_yes": FN_yes,
            "TN_no": TN_no,
            "FP_no": FP_no
        },
        "falsos_positivos_detectados": FP_no,
        "falsos_negativos_detectados": FN_yes,
        "tasa_de_alucinacion_percent": round(tasa_alucinacion, 2),
        "metricas_por_clase": {
            "yes": {
                "precision": round(precision_yes, 4),
                "recall": round(recall_yes, 4),
                "f1": round(f1_yes, 4)
            },
            "no": {
                "precision": round(precision_no, 4),
                "recall": round(recall_no, 4),
                "f1": round(f1_no, 4)
            }
        },
        "macro_promedios": {
            "precision_macro": round(macro_precision, 4),
            "recall_macro": round(macro_recall, 4),
            "f1_macro": round(macro_f1, 4)
        }
    }
#CAMBIAR
def _calcular_hf1_para_un_conjunto(resultados_detallados, arbol, response_key):
    """
    Función auxiliar que calcula hF1 y métricas relacionadas.
    Ahora soporta tanto detecciones únicas como listas de detecciones por imagen.
    La raíz del árbol se excluye de los cálculos de ancestros.
    """
    try:
        root_node = next(n for n, d in arbol.in_degree() if d == 0)
    except StopIteration:

        root_node = None


    scores_hf1 = []
    scores_hp= []
    scores_hr= []
    distancias_lca_emparejadas = []
    total_falsos_positivos = 0
    total_falsos_negativos = 0
    errores_formato = 0
    total_evaluadas = 0
    
    global_intersection_count = 0
    global_total_pred_ancestors = 0
    global_total_gt_ancestors = 0

    for res in resultados_detallados:
        gt = res['ground_truth']
        pred = res.get(response_key, {"error": "Respuesta no encontrada"})

        if pred is None or ('error' in pred and not isinstance(pred, list)):
            errores_formato += 1
            continue
        
        gt_list = gt if isinstance(gt, list) else [gt] 
        pred_list = pred if isinstance(pred, list) else [pred]
        
        pred_list = [p for p in pred_list if p and 'error' not in p]
        if not gt_list or not any(gt_list):
            continue

        try:
            gt_nodes = [f"{g['damage']}_{g['part']}" for g in gt_list] 
            pred_nodes = [f"{p.get('damage')}_{p.get('part')}" for p in pred_list]
            
            valid_gt_nodes = [node for node in gt_nodes if arbol.has_node(node)]
            valid_pred_nodes = [node for node in pred_nodes if arbol.has_node(node)]

            if (not valid_gt_nodes and not valid_pred_nodes) or \
                (not valid_gt_nodes and valid_pred_nodes) or \
                (not valid_pred_nodes and valid_gt_nodes):
                print("Nodo no valido para:", res["imagen"])
                continue 

        except (KeyError, TypeError):
            continue

        gt_pool = []
        for node in valid_gt_nodes: 
            ancestors = nx.ancestors(arbol, node)
            if root_node:
                ancestors.discard(root_node) 
            gt_pool.extend(list(ancestors) + [node])
        pred_pool = []
        for node in valid_pred_nodes:
            ancestors = nx.ancestors(arbol, node)
            if root_node:
                ancestors.discard(root_node)
            pred_pool.extend(list(ancestors) + [node])

        gt_counts = Counter(gt_pool) 
        pred_counts = Counter(pred_pool)

        intersection_count = 0
        for node, count_in_gt in gt_counts.items():
            count_in_pred = pred_counts.get(node, 0)
            intersection_count += min(count_in_gt, count_in_pred) 
        total_pred_ancestors = len(pred_pool)
        total_gt_ancestors = len(gt_pool)
        
        global_intersection_count += intersection_count
        global_total_pred_ancestors += total_pred_ancestors
        global_total_gt_ancestors += total_gt_ancestors
        
        precision_h = intersection_count / total_pred_ancestors if total_pred_ancestors > 0 else 0
        recall_h = intersection_count / total_gt_ancestors if total_gt_ancestors > 0 else 0
        
        hf1 = (2 * precision_h * recall_h) / (precision_h + recall_h) if (precision_h + recall_h) > 0 else 0
        scores_hf1.append(hf1)
        scores_hp.append(precision_h)
        scores_hr.append(recall_h)
        dist_matrix = np.full((len(valid_gt_nodes), len(valid_pred_nodes)), np.inf)
        for i, gt_node in enumerate(valid_gt_nodes):
            for j, pred_node in enumerate(valid_pred_nodes):
                if gt_node == pred_node:
                    dist_matrix[i, j] = 0
                else:
                    try:
                        lca = nx.lowest_common_ancestor(arbol, gt_node, pred_node)
                        dist = nx.shortest_path_length(arbol, lca, gt_node) + nx.shortest_path_length(arbol, lca, pred_node)
                        dist_matrix[i, j] = dist
                    except nx.NetworkXNoPath:
                        continue
        
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched_gt_indices = set(row_ind)
        matched_pred_indices = set(col_ind)
        
        for i, j in zip(row_ind, col_ind):
            dist = dist_matrix[i, j]
            if dist > 0:
                distancias_lca_emparejadas.append(dist_matrix[i, j])

        fp = len(valid_pred_nodes) - len(matched_pred_indices)
        fn = len(valid_gt_nodes) - len(matched_gt_indices)
        total_falsos_positivos += fp
        total_falsos_negativos += fn
        total_evaluadas+=1

    avg_hf1 = sum(scores_hf1) / len(scores_hf1) if scores_hf1 else 0
    avg_hp = sum(scores_hp) / len(scores_hp) if scores_hp else 0 
    avg_hr = sum(scores_hr) / len(scores_hr) if scores_hr else 0
    avg_lca_distance = sum(distancias_lca_emparejadas) / len(distancias_lca_emparejadas) if distancias_lca_emparejadas else 0
    
    global_precision_h = global_intersection_count / global_total_pred_ancestors if global_total_pred_ancestors > 0 else 0
    global_recall_h = global_intersection_count / global_total_gt_ancestors if global_total_gt_ancestors > 0 else 0
    hF1_global = (2 * global_precision_h * global_recall_h) / (global_precision_h + global_recall_h) if (global_precision_h + global_recall_h) > 0 else 0
    
    return {
        "respuestas_validas": total_evaluadas,
        "errores_de_formato": errores_formato,
        "hP_promedio": round(avg_hp, 4),  
        "hR_promedio": round(avg_hr, 4),
        "hF1_promedio": round(avg_hf1, 4),
        "hP_global": round(global_precision_h, 4),
        "hR_global": round(global_recall_h, 4),
        "hF1_global": round(hF1_global,4),
        "distancia_LCA_promedio_en_errores": round(avg_lca_distance, 2),
        "total_falsos_positivos": total_falsos_positivos,
        "total_falsos_negativos": total_falsos_negativos,
    }
    #Meter en params los returns
def calcular_metricas_jerarquicas(resultados_detallados, arbol):
    """Calcula las métricas jerárquicas y la ganancia multimodal."""
    
    stats_multimodal = _calcular_hf1_para_un_conjunto(resultados_detallados, arbol, 'respuesta_normalizada')
    
    #stats_text_only = _calcular_hf1_para_un_conjunto(resultados_detallados, arbol, 'respuesta_normalizada_text_only')

    #ganancia_hf1_promedio = stats_multimodal['hF1_promedio'] - stats_text_only['hF1_promedio']
    #ganancia_hf1_global = stats_multimodal['hF1_global'] - stats_text_only['hF1_global']

    return {
        "metrica_principal": "F1-Score Jerárquico (hF1) y Ganancia Multimodal",
        "total_preguntas_evaluadas": len(resultados_detallados),

        # "ganancia_multimodal": {
        #     "sobre_hF1_promedio": round(ganancia_hf1_promedio, 4),
        #     "sobre_hF1_global": round(ganancia_hf1_global, 4)
        # },
        "metricas_multimodal": stats_multimodal
    }

def calcular_metricas_pds(resultados_detallados, arbol, baseline_hf1):
    """Calcula la caída de rendimiento hF1 por nivel para la tarea PDS."""
    resultados_por_nivel = {}
    for res in resultados_detallados:
        nivel = res.get('nivel')
        if nivel is not None:
            if nivel not in resultados_por_nivel:
                resultados_por_nivel[nivel] = []
            resultados_por_nivel[nivel].append(res)

    metricas_pds = {
        "metrica_principal": "Caída de Rendimiento hF1 por Nivel (PDS)",
        "hF1_base_original": baseline_hf1,
        "rendimiento_por_nivel": {}
    }

    for nivel, resultados_nivel in sorted(resultados_por_nivel.items()):
        if not resultados_nivel:
            continue

        stats_nivel = _calcular_hf1_para_un_conjunto(resultados_nivel, arbol, 'respuesta_normalizada')
        # La métrica principal para PDS es el hF1 global, que es más robusto.
        hf1_nivel_global = stats_nivel['hF1_global']
        
        caida_rendimiento = 0
        if baseline_hf1 > 0:
            caida_rendimiento = ((baseline_hf1 - hf1_nivel_global) / baseline_hf1) * 100
        else:
            caida_rendimiento = 0.0

        metricas_pds["rendimiento_por_nivel"][f"Nivel_{nivel}"] = {
            "hF1_global_nivel": hf1_nivel_global,
            "caida_rendimiento_percent": round(caida_rendimiento, 2)
        }
    
    return metricas_pds

def despachar_calculo_metricas(task_name, resultados, arbol_taxonomia=None, metricas_base=None):
    if task_name == "Hallucination":
        return calcular_metricas_alucinacion(resultados)
    
    if task_name == "hF1":
        if arbol_taxonomia is None:
            return {"error": "El árbol de taxonomía es necesario para calcular las métricas jerárquicas."}
        return calcular_metricas_jerarquicas(resultados, arbol_taxonomia)

    if task_name == "PDS":
        if arbol_taxonomia is None:
            return {"error": "El árbol de taxonomía es necesario para la tarea PDS."}
        if metricas_base and 'hF1' in metricas_base and 'metricas' in metricas_base['hF1']:
            # Usamos el hF1_global del multimodal como baseline, es más robusto
            baseline_hf1 = metricas_base['hF1']['metricas']['metricas_multimodal'].get('hF1_global', 0)
            return calcular_metricas_pds(resultados, arbol_taxonomia, baseline_hf1)
        else:
            return {"error": "Las métricas de la tarea 'hF1' base son necesarias para calcular PDS y no se encontraron. Asegúrate de que 'hF1' se procese primero."}

    print(f"Advertencia: No hay una función de métricas definida para la tarea '{task_name}'.")
    return {"error": f"Métricas no definidas para la tarea '{task_name}'"}