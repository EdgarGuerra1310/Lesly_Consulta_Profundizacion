import os
import json
import random
from typing import List, Dict
import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from docx import Document
import tiktoken

# ==============================
# ⚙️ CONFIGURACIÓN INICIAL
# ==============================

load_dotenv()

app = FastAPI(title="Chat Cursos de Profundización")

# Templates
templates = Jinja2Templates(directory="templates")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró la API key de OpenAI. Verifica tu .env")

openai_client = OpenAI(api_key=api_key)


# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://91.99.108.245:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Modelo de embeddings
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
embedding_model = SentenceTransformer(EMBED_MODEL)
print(f"✅ Modelo cargado. Dimensión: {embedding_model.get_sentence_embedding_dimension()}")

# ==============================
# 🗄️ BASE DE DATOS (PostgreSQL)
# ==============================

DB_CONFIG = {
    "dbname": "proyectos_ia",
    "user": "postgres",
    "password": "adm",
    "host": "localhost",
    "port": "5432",
}

def get_db_connection():
    return psycopg2.connect(
        **DB_CONFIG,
        options="-c client_encoding=UTF8"
    )

def crear_tabla_interacciones():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS public.interacciones_cursos (
            id SERIAL PRIMARY KEY,
            curso TEXT NOT NULL,
            usuario TEXT NOT NULL,
            mensaje TEXT NOT NULL,
            respuesta TEXT NOT NULL,
            tokens_mensaje INT DEFAULT 0,
            tokens_respuesta INT DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# ==============================
# 📊 MODELOS PYDANTIC
# ==============================

class ChatRequest(BaseModel):
    message: str
    usuario: str
    collection_name: str

class QuestionRequest(BaseModel):
    collection_name: str
    num_questions: int = 10

# ==============================
# 🧠 FUNCIONES DE EMBEDDING Y BÚSQUEDA
# ==============================

embedding_cache = {}
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

def contar_tokens(texto: str) -> int:
    """Cuenta los tokens de un texto según el modelo."""
    return len(encoding.encode(texto))

def embed_query(query: str) -> np.ndarray:
    """Genera embedding para una consulta con cache usando sentence-transformers."""
    if query in embedding_cache:
        return embedding_cache[query]

    # Generar embedding con el modelo
    embedding = embedding_model.encode(query, normalize_embeddings=True)
    embedding = np.array(embedding, dtype="float32")
    
    embedding_cache[query] = embedding
    return embedding


def search_qdrant(collection_name: str, query: str, k: int = 10) -> List[Dict]:
    """
    Busca en Qdrant y devuelve los k chunks más similares.
    """
    query_vector = embed_query(query).tolist()
    
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k
    )
    
    results = []
    for hit in search_result:
        payload = hit.payload
        results.append({
            "text": payload.get("text", ""),
            "filename": payload.get("filename", "Documento no especificado"),
            "chunk": payload.get("chunk", 0),
            "total_chunks": payload.get("total_chunks", 0),
            "total_pages": payload.get("total_pages", 0),
            "score": hit.score
        })
    
    return results


def get_random_chunks(collection_name: str, n: int = 5) -> List[Dict]:
    """Obtiene chunks aleatorios de una colección para generar preguntas."""
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=n * 3,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]
        
        if len(points) > n:
            points = random.sample(points, n)
        
        results = []
        for point in points:
            payload = point.payload
            results.append({
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "chunk": payload.get("chunk", 0)
            })
        
        return results
    except Exception as e:
        print(f"Error obteniendo chunks aleatorios: {e}")
        return []

# ==============================
# 🛠️ FUNCIÓN DE NORMALIZACIÓN UTF-8
# ==============================

def normalizar_texto(texto: str) -> str:
    """Asegura que cualquier texto sea UTF-8 válido."""
    if not texto:
        return texto
    
    try:
        if isinstance(texto, str):
            return texto.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        elif isinstance(texto, bytes):
            return texto.decode('utf-8', errors='replace')
        else:
            return str(texto)
    except Exception as e:
        print(f"Error normalizando texto: {e}")
        try:
            return texto.encode('ascii', errors='ignore').decode('ascii')
        except:
            return str(texto)

# ==============================
# 💬 GENERADOR DE PREGUNTAS
# ==============================

def generate_recommended_questions(text: str, n: int = 5) -> Dict[str, List[str]]:
    """Genera preguntas pedagógicas clasificadas por tipo."""
    prompt = f"""
A partir del siguiente contenido del curso, genera preguntas pedagógicas
clasificadas en los siguientes tipos:

1. Preguntas de contenido:
   - Buscan comprender, aclarar o recuperar información conceptual.
   - Se enfocan en definiciones, características, clasificaciones o explicaciones.
   - Ejemplo: ¿Qué es la regulación emocional según el CNEB?

2. Preguntas de contexto:
   - Buscan situar el contenido en una realidad educativa concreta.
   - Consideran nivel educativo, territorio, tipo de institución o características del estudiante.
   - Ejemplo: ¿Qué estrategias usar con estudiantes de nivel inicial?

3. Preguntas de reflexión:
   - Estimulan el análisis crítico y la autoexploración.
   - No buscan respuestas cerradas.
   - Ejemplo: ¿Qué otras estrategias puedo usar para mejorar el clima del aula?

Contenido del curso:
\"\"\"
{text}
\"\"\"

Genera exactamente {n} preguntas por cada tipo.

Devuelve SOLO un JSON válido con la siguiente estructura:

{{
  "contenido": ["pregunta 1", "pregunta 2", "..."],
  "contexto": ["pregunta 1", "pregunta 2", "..."],
  "reflexion": ["pregunta 1", "pregunta 2", "..."]
}}

No incluyas explicaciones ni texto adicional fuera del JSON.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        questions = json.loads(raw)
    except Exception as e:
        print(f"Error generando preguntas: {e}")
        questions = {
            "contenido": [],
            "contexto": [],
            "reflexion": []
        }

    return questions

# ==============================
# 📄 PROMPT BASE
# ==============================

def leer_prompt_desde_word(path_docx: str) -> str:
    """Lee el prompt base desde un archivo Word."""
    try:
        doc = Document(path_docx)
        texto = "\n".join([p.text for p in doc.paragraphs])
        texto_limpio = texto.strip()
        return texto_limpio.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        print(f"Error leyendo prompt base: {e}")
        return """Eres un asistente pedagógico especializado en educación.

Contexto del curso:
{context}

Pregunta del usuario:
{question}

Instrucciones:
- Responde de manera clara y concisa basándote en el contexto proporcionado
- Usa ejemplos prácticos cuando sea apropiado
- Si la información no está en el contexto, indícalo claramente
- Mantén un tono profesional y amable
- Enfócate en aplicaciones prácticas para docentes"""

PROMPT_TEMPLATE = leer_prompt_desde_word("prompt_base.docx")


def formatear_chunk_para_contexto(chunk: Dict) -> str:
    """Formatea un chunk para incluirlo en el contexto."""
    filename = chunk.get("filename", "Documento no especificado")
    chunk_num = chunk.get("chunk", 0)
    total_chunks = chunk.get("total_chunks", 0)
    text = chunk.get("text", "")

    ubicacion = f"Chunk {chunk_num}/{total_chunks}" if total_chunks > 0 else f"Chunk {chunk_num}"

    return f"""[{filename} – {ubicacion}]
{text}
"""

# ==============================
# 🌐 RUTAS DE LA API
# ==============================

@app.on_event("startup")
async def startup_event():
    """Inicializa recursos al arrancar la aplicación."""
    crear_tabla_interacciones()
    print("🚀 Iniciando aplicación...")
    print("✅ Aplicación lista")


@app.get("/asistente_profundizacion", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal del chat."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/collections")
async def get_collections():
    """Obtiene la lista de colecciones disponibles en Qdrant."""
    try:
        collections = qdrant_client.get_collections()
        
        result = []
        for collection in collections.collections:
            try:
                count_result = qdrant_client.count(
                    collection_name=collection.name,
                    exact=True
                )
                
                result.append({
                    "name": collection.name,
                    "points_count": count_result.count,
                    "vectors_count": count_result.count
                })
            except Exception as e:
                print(f"Advertencia al obtener info de {collection.name}: {e}")
                try:
                    scroll_result = qdrant_client.scroll(
                        collection_name=collection.name,
                        limit=1,
                        with_payload=False,
                        with_vectors=False
                    )
                    result.append({
                        "name": collection.name,
                    })
                except:
                    result.append({
                        "name": collection.name
                    })
        
        return {"collections": result}
    except Exception as e:
        print(f"Error obteniendo colecciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommended_questions")
async def get_recommended_questions(request: QuestionRequest):
    """Genera preguntas recomendadas basadas en una colección."""
    try:
        chunks = get_random_chunks(request.collection_name, n=5)
        
        if not chunks:
            return {
                "contenido": ["No hay contenido disponible en esta colección."],
                "contexto": [],
                "reflexion": []
            }
        
        combined_text = " ".join([chunk["text"][:500] for chunk in chunks])
        
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]
        
        questions = generate_recommended_questions(
            combined_text, 
            n=request.num_questions
        )
        
        return questions
    except Exception as e:
        print(f"Error generando preguntas recomendadas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Procesa un mensaje del chat y devuelve la respuesta, guardando tokens."""
    conn = None 
    cursor = None
    
    try:
        relevant_chunks = search_qdrant(
            collection_name=request.collection_name,
            query=request.message,
            k=5
        )
        
        if not relevant_chunks:
            return {
                "answer": "Lo siento, no encontré información relevante en la colección seleccionada. Por favor, intenta reformular tu pregunta."
            }
        
        context_parts = [
            formatear_chunk_para_contexto(chunk)[:800] 
            for chunk in relevant_chunks
        ]
        context = "\n\n".join(context_parts)
        
        prompt = PROMPT_TEMPLATE.format(
            context=context, 
            question=request.message
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4
        )
        answer = response.choices[0].message.content.strip()

        # Contar tokens reales: prompt completo enviado a OpenAI
        tokens_mensaje = contar_tokens(prompt)
        tokens_respuesta = contar_tokens(answer)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            collection_safe = normalizar_texto(request.collection_name)
            usuario_safe = normalizar_texto(request.usuario)
            mensaje_safe = normalizar_texto(request.message)
            respuesta_safe = normalizar_texto(answer)
            
            cursor.execute("""
                INSERT INTO public.interacciones_cursos
                (curso, usuario, mensaje, respuesta, tokens_mensaje, tokens_respuesta)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                collection_safe,
                usuario_safe,
                mensaje_safe,
                respuesta_safe,
                tokens_mensaje,
                tokens_respuesta
            ))
            conn.commit()
            print(f"✅ Interacción guardada: {usuario_safe} - {tokens_mensaje + tokens_respuesta} tokens")
            
        except Exception as e:
            print(f"⚠️ Error guardando interacción: {e}")
            print(f"   Collection: {repr(request.collection_name[:50])}")
            print(f"   Usuario: {repr(request.usuario)}")
            print(f"   Mensaje: {repr(request.message[:50])}")
            print(f"   Respuesta: {repr(answer[:50])}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        return {
            "answer": answer,
            "tokens_mensaje": tokens_mensaje,
            "tokens_respuesta": tokens_respuesta
        }
        
    except Exception as e:
        print(f"Error en chat: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando la consulta: {str(e)}"
        )
    

# ==============================
# 🚀 EJECUTAR SERVIDOR
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)