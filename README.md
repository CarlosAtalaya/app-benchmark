**Explicacion de los script:**

- config.py: gestiona la carga de la configuración de la aplicación desde el archivo config.yaml. Centraliza parámetros importantes como endpoints, directorios y modelos. (Ya no)

- utils.py: Aqui se encuentran las funciones para aplicar distorsiones visuales a las imágenes, cargarlas en formato base64 y construir el grafo a partir de una taxonomía de daños definida

- models.py: gestiona la comunicación con los modelos de IA a través de su API. Contiene funciones para enviar prompts a modelos de visión, además de la herramienta para normalizar y estructurar las respuestas brutas en formato json.

- metrics.py: implementa la lógica para calcular las métricas de evaluación del modelo. Contiene funciones para medir la tasa de alucinación, el hF1 que considera la taxonomía de daños, y la PDS al introducir distorsiones en las imágenes.

- reporting.py: imprime los resultados por pantalla y genera el reporte final con las métricas calculadas y los resultados individuales detallados.

- main.py: script principal del proceso de evaluación. Carga la configuración, itera sobre las imágenes y sus archivos JSON de tareas, usa los modelos de IA para obtener respuestas y finalmente utiliza los scripts de métricas para calcular y generar un informe de resultados completo.

-run_evaluations.py: Para poder ejecutar el analisis de diversos modelos de forma consecutiva

-front.py: Ejecuta el streamlit con la herramienta
**Configuración (config.yaml):**

-modelo_a_evaluar: Modelo de ollama que queremos evaluar (ollama en este caso o cualquier endpoint)

-modelos_a_evaluar_en_lote: Esta funcion es para la ejecucion del run_evaluations.py Aqui ponemos los modelos que queremos analizar en este formato:

modelos_a_evaluar_en_lote:
- qwen2.5vl:latest
- llava:7b
- minicpm-v:8b

-directorio_imagenes: carpeta donde se encuentra el dataset para hacer la evaluacion

- tareas_a_ejecutar: 
    - hf1: Calculo del hf1 global y promedio, asi como el LCD, el hP, hR y total de falsos positivos y falsos negativos
    - Hallucination: Plantea preguntas de si o no al modelo y comprueba su tasa de alucinación. Actualmente esta planteado como analisis de filtro inicial para determinar si hay daño o no en la imagen (no filtra realmente, pero se plantea como eso)
    - PDS: Realiza el mismo estudio que para el hf1 pero con imagenes degradas y realiza una comparación con la variación de este por nivel. El nivel se puede configurar en cada json de forma individual o en el generador de prompts

ollama_endpoint: endpoint de donde se usa el modelo, pone ollama, pero se puede usar cualquiera 

output_file: nombre y dirección de donde se quiere almacenar el fichero json con el reporte completo

**JSONS**

- taxonomia.json: Contiene la taxonomia de daños. Es necesaria para generar el grafo a la hora de calcular las metricas

- resultados_evaluacion.json: json final con toda la información 

**Explicación del flujo de uso:**


1. Creacion del entorno:

'''
pip install -r requirements.txt
'''

2. Ejecutar generador_prompt_task.py

Este script te permite atribuirle un json con los prompt para cada task a cada imagen. No pone el GT a las imagenes (por razones obvias, esto debe hacerse a mano), pero si pone una plantilla por defecto. En el caso de que el json ya este creado, mantendra el GT original, cambiando solo los prompts (esto permite cambiar todos los promtps de todos los jsons a la vez sin alterear el GT)

3. Ejecutar el main.py


**Ejemplo del formato que debe ir en los json que acompañan cada imagen**

{
  "Hallucination": [
    {
      "prompt": "Observando únicamente la imagen, ¿hay daños en el coche? Responde solo con 'Sí' o 'No'.",
      "ground_truth": "No"
    }
  ],
  "hF1": [
    {
      "prompt": "Eres un perito experto en daños de vehículos. Analiza la imagen adjunta y clasifica los daños que veas. Devuelve un objeto JSON o una lista con objetos JSON por cada tipo de daño que encuentres con esta estructura (sin texto adicional, sin explicaciones):\n\n{\n\"daño\": \"VALOR_DE_LA_LISTA_DE_DAÑOS\",\n\"pieza\": \"VALOR_DE_LA_LISTA_DE_PIEZAS\",\n \"severidad\": \"VALOR_DE_LA_LISTA_DE_SEVERIDADES\"\n}\n\nSi no observas ningún daño en la imagen, responde únicamente con el texto 'No hay daño'.\nSi no puedes determinar el daño, la pieza o la severidad, responde únicamente con el texto 'No lo sé'.\n\nUsa exclusivamente los valores de cada lista para el JSON:\n\nDaños válidos: \"Arañazo\",\"Abolladura\",\"Barniz degradado\",\"Fisura\",\"Pieza Fracturada\",\"Pieza faltante\",\"Pieza desviada\",\"No hay daño\",\"Desconocido\"\n\nPiezas válidas: \"Parachoques delantero\",\"Parachoaches trasero\",\"Capó\",\"Portón trasero\",\"Puerta delantera izquierda\",\"Puerta delantera derecha\",\"Puerta trasera derecha\",\"Puerta trasera izquierda\",\"Aleta delantera izquierda\",\"Aleta delantera derecha\",\"Aleta trasera izquierda\",\"Aleta trasera derecha\",\"Ninguna\"\n\nSeveridades válidas: \"Leve\",\"Grave\"\n\nNo inventes valores fuera de las listas. La única salida permitida es el objeto JSON o, en los casos indicados, el texto 'No hay daño' o 'No lo sé'.",
      "ground_truth": {
        "daño": "No hay daño",
        "pieza": "Ninguna",
        "severidad": "N/A"
      }
    }
  ],
  "PDS": [
    {
      "prompt": "Eres un perito experto en daños de vehículos. Analiza la imagen adjunta y clasifica los daños que veas. Devuelve un objeto JSON o una lista con objetos JSON por cada tipo de daño que encuentres con esta estructura (sin texto adicional, sin explicaciones):\n\n{\n\"daño\": \"VALOR_DE_LA_LISTA_DE_DAÑOS\",\n\"pieza\": \"VALOR_DE_LA_LISTA_DE_PIEZAS\",\n \"severidad\": \"VALOR_DE_LA_LISTA_DE_SEVERIDADES\"\n}\n\nSi no observas ningún daño en la imagen, responde únicamente con el texto 'No hay daño'.\nSi no puedes determinar el daño, la pieza o la severidad, responde únicamente con el texto 'No lo sé'.\n\nUsa exclusivamente los valores de cada lista para el JSON:\n\nDaños válidos: \"Arañazo\",\"Abolladura\",\"Barniz degradado\",\"Fisura\",\"Pieza Fracturada\",\"Pieza faltante\",\"Pieza desviada\",\"No hay daño\",\"Desconocido\"\n\nPiezas válidas: \"Parachoques delantero\",\"Parachoaches trasero\",\"Capó\",\"Portón trasero\",\"Puerta delantera izquierda\",\"Puerta delantera derecha\",\"Puerta trasera derecha\",\"Puerta trasera izquierda\",\"Aleta delantera izquierda\",\"Aleta delantera derecha\",\"Aleta trasera izquierda\",\"Aleta trasera derecha\",\"Ninguna\"\n\nSeveridades válidas: \"Leve\",\"Grave\"\n\nNo inventes valores fuera de las listas. La única salida permitida es el objeto JSON o, en los casos indicados, el texto 'No hay daño' o 'No lo sé'.",
      "ground_truth": {
        "daño": "No hay daño",
        "pieza": "Ninguna",
        "severidad": "N/A"
      },
      "nivel": 2
    }
  ]
}

**Uso de Streamlit**

Para ejecutar streamlit solo es necesario ejecutar el comando:

streamlit run front.py --server.port 8502 (el puerto puede ser el que se desee)

Una vez hecho esto, a la izquierda aparecen estas secciones:

"Selecciona el directorio de imagenes": 

Debe ser la carpeta con las imagenes y los json dentro, contenida en la misma ruta donde se ejecuta la aplicacion

Tareas a Ejecutar:

Indicamos que tareas queremos analizar. Al final del proceso se generan los resportes por modelo en la carpeta "reportes"

## Nueva Funcionalidad: RAG Multimodal (Retrieval-Augmented Generation)

Esta versión introduce un pipeline de Generación Aumentada por Recuperación (RAG) que potencia la capacidad del modelo VLM proporcionándole "memoria visual" de casos históricos similares.

### Funcionamiento de la Arquitectura RAG

El sistema no se limita a pasar la imagen al modelo. Antes de la inferencia, ejecuta un pipeline de visión computacional avanzado:

1.  **Segmentación Semántica (SAM 3)**
    Utiliza el modelo Segment Anything Model 3 para generar una máscara precisa del vehículo, eliminando el "ruido" del fondo (calles, otros coches, personas) que suele confundir a los modelos generales.

2.  **Análisis Granular (Grid Crops)**
    La imagen limpia se divide en una rejilla de recortes de alta resolución (336x336px). Esto permite detectar micro-daños (arañazos leves, piquetes) que se perderían al redimensionar la imagen completa.

3.  **Búsqueda Vectorial (MetaCLIP + FAISS)**
    Cada recorte se convierte en un vector matemático (embedding) utilizando MetaCLIP. El sistema busca en una base de datos vectorial local (`vector_indices`) casos históricos visualmente similares (ej: "arañazo en parachoques blanco").

4.  **Inyección de Contexto**
    Si se encuentran similitudes con alta confianza, se construye un resumen textual (ej: "Referencia visual: Se han detectado patrones compatibles con 'Abolladura leve' en la aleta delantera con un 85% de similitud"). Este contexto se inyecta dinámicamente en el prompt del usuario, guiando al VLM para que su respuesta sea más técnica y precisa.