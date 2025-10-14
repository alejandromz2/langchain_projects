# 1. Configuración Inicial
from langchain_core.runnables import RunnableLambda, RunnableParallel

from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model='gpt-4o-mini', temperature='0')

# 2. Preprocesador de Texto

def preprocess_text(text):
    """Limpia el texto eliminando espacios extras y limitando longitud"""
    # Pista: usa .strip() para eliminar espacios
    # Pista: limita a 500 caracteres con slicing [:500]
    return text.strip()[:500] # ¡Completa aquí!
 
# Convertir la función en un Runnable
preprocessor = RunnableLambda(preprocess_text)


#3. Generador de Resúmenes
def generate_summary(text):
    """Genera un resumen conciso del texto"""
    prompt = f"""Resume en una sola oración considerando la idea
    principal del texto. Maneja oraciones de menos de 500 palabras : {text}"""
    response = llm.invoke(prompt)
    return response.content

summary_branch = RunnableLambda(generate_summary)

# 4. Analizador de Sentimientos

def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}
    
    Texto: {text}"""
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}
    
sentiment_branch = RunnableLambda(analyze_sentiment)

# 5. Función de Combinación

def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }

merger = RunnableLambda(merge_results)

# 6. Función de Procesamiento Principal

paralel_analysis = RunnableParallel({
    "resumen":summary_branch,
    "sentimiento_data":sentiment_branch
})
 

# 7. Construcción de la Cadena Final
chain = preprocessor | paralel_analysis | merger

# Prueba con diferentes textos
textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]
 
for texto in textos_prueba:
    resultado = chain.invoke(texto)
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-" * 50)