# models.py
import requests
import json
import re

import google.generativeai as genai
from openai import OpenAI
import base64
from perplexity import Perplexity

SESSION = requests.Session()
def consultar_modelo_vlm(prompt, image_b64, model_name, temperature, top_k, endpoint, gestor, api_key=None):
    """
    Realiza una consulta a un modelo VLM, adaptándose al gestor especificado.
    La api_key es requerida para los gestores 'openai' y 'gemini'.
    """
    headers = {"Content-Type": "application/json"}
    payload = {}

    if gestor == 'ollama':
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "images": [image_b64],
            "options": {"temperature": temperature, "top_k": top_k}
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data.get("response", "").strip()
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            eval_tokens = response_data.get("eval_count", 0)
            
            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo VLM ({gestor}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Detalle del error: {e.response.text}")

    elif gestor == 'lm_studio':
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": 2048
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            prompt_tokens = response_data['usage']['prompt_tokens']
            eval_tokens = response_data['usage']['completion_tokens']

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo VLM ({gestor}): {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error al procesar la respuesta de {gestor}: {e} - Respuesta recibida: {response_data}")
            return None
        
    elif gestor == 'qwen':
        # Este es el payload que tu api_server.py modificado espera
        payload = {
            "image_b64": image_b64,
            "text": prompt
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data.get('response', '').strip()

            # Tu servidor no devuelve el conteo de tokens, así que los ponemos a 0.
            # Esto es una limitación de tu servidor personalizado.
            return {
                "response": response_text,
                "prompt_tokens": 0,
                "eval_tokens": 0
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo VLM ({gestor}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Detalle del error: {e.response.text}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error al procesar la respuesta de {gestor}: {e} - Respuesta recibida: {response_data}")
            return None
        
    elif gestor == 'qwen3':
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            prompt_tokens = response_data['usage']['prompt_tokens']
            eval_tokens = response_data['usage']['completion_tokens']

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo VLM ({gestor}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Detalle del error: {e.response.text}")
            return None
        
    elif gestor == 'vllm':
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 5000
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            prompt_tokens = response_data['usage']['prompt_tokens']
            eval_tokens = response_data['usage']['completion_tokens']

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo VLM ({gestor}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Detalle del error: {e.response.text}")
            return None


    elif gestor == 'openai':
        if not api_key:
            print("Error: Se requiere una API key para el gestor 'openai'.")
            return None
        try:
            client_openai = OpenAI(api_key=api_key) 
            response = client_openai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                temperature=temperature
            )
            response_text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            eval_tokens = response.usage.completion_tokens
            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except Exception as e:
            print(f"Error al conectar con el modelo VLM (OpenAI): {e}")
            return None

    elif gestor == 'nova':
        if not api_key:
            print("Error: Se requiere una API key para el gestor 'nova'.")
            return None

        try:
            

            client_nova = Perplexity(api_key=api_key)

            response = client_nova.responses.create(
                model=model_name,  
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=None,
                temperature=temperature
            )

            response_text = response.output_text.strip()
            prompt_tokens = response.usage.input_tokens
            eval_tokens = response.usage.output_tokens

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }

        except Exception as e:
            print(f"Error al conectar con el modelo VLM (NOVA/Perplexity): {e}")
            return None

    elif gestor == 'gemini':
        if not api_key:
            print("Error: Se requiere una API key para el gestor 'gemini'.")
            return None
        try:
            genai.configure(api_key=api_key) 
            model = genai.GenerativeModel(model_name)
            image_blob = {"mime_type": "image/jpeg", "data": base64.b64decode(image_b64)}
            
            response = model.generate_content(
                [prompt, image_blob],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    top_k=top_k
                )
            )
            
            prompt_tokens = model.count_tokens([prompt, image_blob]).total_tokens
            eval_tokens = model.count_tokens(response.text).total_tokens

            return {
                "response": response.text.strip(),
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except Exception as e:
            print(f"Error al conectar con el modelo VLM (Gemini): {e}")
            return None
            
    else:
        print(f"Error: Gestor '{gestor}' no soportado.")
        return None

def consultar_modelo_text_only(prompt, model_name, temperature, top_k, endpoint, gestor, api_key=None):
    """
    Realiza una consulta a un modelo de solo texto, adaptándose al gestor.
    La api_key es requerida para los gestores 'openai' y 'gemini'.
    """
    headers = {"Content-Type": "application/json"}
    payload = {}

    if gestor == 'ollama':
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "top_k": top_k}
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data.get("response", "").strip()
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            eval_tokens = response_data.get("eval_count", 0)

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo Text-Only ({gestor}): {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error al procesar la respuesta de {gestor}: {e} - Respuesta recibida: {response_data}")
            return None

    elif gestor == 'lm_studio':
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2048,
            "top_k": top_k
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            prompt_tokens = response_data['usage']['prompt_tokens']
            eval_tokens = response_data['usage']['completion_tokens']

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo Text-Only ({gestor}): {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error al procesar la respuesta de {gestor}: {e} - Respuesta recibida: {response_data}")
            return None
    elif gestor == 'vllm':
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2048,
            "top_k": top_k
        }
        try:
            response = SESSION.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            prompt_tokens = response_data['usage']['prompt_tokens']
            eval_tokens = response_data['usage']['completion_tokens']

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con el modelo Text-Only ({gestor}): {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error al procesar la respuesta de {gestor}: {e} - Respuesta recibida: {response_data}")
            return None
            
    elif gestor == 'openai':
        if not api_key:
            print("Error: Se requiere una API key para el gestor 'openai'.")
            return None
        try:
            client_openai = OpenAI(api_key=api_key)
            response = client_openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2048
            )
            response_text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            eval_tokens = response.usage.completion_tokens
            
            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except Exception as e:
            print(f"Error al conectar con el modelo Text-Only (OpenAI): {e}")
            return None

    elif gestor == 'gemini':
        if not api_key:
            print("Error: Se requiere una API key para el gestor 'gemini'.")
            return None
        try:
            genai.configure(api_key=api_key) 
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    top_k=top_k
                )
            )
            
            prompt_tokens = model.count_tokens(prompt).total_tokens
            eval_tokens = model.count_tokens(response.text).total_tokens

            return {
                "response": response.text.strip(),
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens
            }
        except Exception as e:
            print(f"Error al conectar con el modelo Text-Only (Gemini): {e}")
            return None
            
    else:
        print(f"Error: Gestor '{gestor}' no soportado.")
        return None

def normalizar_respuesta(texto_raw, ground_truth):
    if not texto_raw:
        return {"error": "Respuesta vacía"}

    texto_limpio_raw = texto_raw.lower().strip()
    if "no damage" in texto_limpio_raw:
        return {"damage": "No damage","part": "None"}
    if "i don't know" in texto_limpio_raw or "desconocido" in texto_limpio_raw:
        return {"damage": "Unknowk","part": "None"}

    if isinstance(ground_truth, list):
        try:
            json_match = re.search(r'```json\s*(\[.*?\])\s*```|(\[.*?\])', texto_raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
                return json.loads(json_str)
            else:
                json_objects = re.findall(r'(\{.*?\})', texto_raw, re.DOTALL)
                if json_objects:
                    return [json.loads(obj) for obj in json_objects]
                return {"error": "No se encontró una lista de JSONs válida en la respuesta."}
        except json.JSONDecodeError:
            return {"error": f"Error al decodificar la lista de JSONs: {texto_raw}"}

    if isinstance(ground_truth, dict):
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', texto_raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
                return json.loads(json_str)
            else:
                return {"error": "No se encontró un JSON válido en la respuesta."}
        except json.JSONDecodeError:
            return {"error": f"Error al decodificar JSON: {texto_raw}"}

    texto_limpio = texto_raw.lower().strip().replace(".", "")
    if isinstance(ground_truth, str) and ground_truth.lower() in ['yes','no']:
        if texto_limpio.startswith('yes') or texto_limpio.startswith('yes'):
            return 'Yes'
        if texto_limpio.startswith('no'):
            return 'No'
        return "Indefinido"
    
    return texto_raw.strip().capitalize()