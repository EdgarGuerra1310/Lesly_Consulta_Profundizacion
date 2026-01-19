#  💬 Chat BIAE 

Una aplicación de chat basada en FastAPI que aprovecha la base de datos vectorial Qdrant y los modelos de lenguaje de OpenAI para proporcionar conversaciones inteligentes y contextualmente relevantes basadas en materiales de cursos PDF.

## 🎯 Características

- **Búsqueda basada en vectores**: Utiliza la base de datos Qdrant para búsquedas de similitud eficientes en materiales de cursos
- **Respuestas impulsadas por IA**: Integra la API de OpenAI para generar respuestas inteligentes
- **Embeddings semánticos**: Utiliza transformadores de oraciones (BAAI/bge-base-en-v1.5) para generación de incrustaciones
- **Interfaz web**: Interfaz HTML hermosa y responsiva para fácil interacción
- **Soporte de documentos**: Procesa e indexa documentos de Word (.docx) para contenido del curso
- **API RESTful**: Backend basado en FastAPI para integración perfecta

## 💾 Pila tecnológica

- **Backend**: FastAPI, Uvicorn
- **Base de datos vectorial**: Qdrant
- **ML/IA**: 
  - Sentence Transformers (para incrustaciones)
  - API de OpenAI (para generación de texto)
- **Frontend**: HTML5 con JavaScript vanilla
- **Lenguaje**: Python 3.x

## 📋 Requisitos previos

- Python 3.8+
- Instancia de Qdrant ejecutándose (configurada mediante variables de entorno)
- Clave de API de OpenAI
- Entorno virtual (recomendado)

## 🚀 Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <repository-url>
   cd chat_biae
   ```

2. **Crear un entorno virtual**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
   Crea un archivo `.env` en la raíz del proyecto:
   ```
   QDRANT_URL=http://91.99.108.245:6333
   QDRANT_API_KEY=tu_clave_api_aqui
   API_KEY_OPENAI=tu_clave_openai_aqui
   ```

##  ⭐ Ejecutar la aplicación

Inicia el servidor de desarrollo:

```bash
python main.py
```

O ejecuta con Uvicorn directamente:

```bash
uvicorn main:app --reload
```

La aplicación estará disponible en `http://localhost:8000`

## 📁 Estructura del proyecto

```
chat_biae/
├── main.py              # Aplicación principal FastAPI
├── requirements.txt     # Dependencias de Python
├── README.md           # Este archivo
├── .env                # Variables de entorno (crear localmente)
└── templates/
    └── index.html      # Interfaz web
```

## 🔧 Uso

1. **Acceder a la interfaz web**: Abre tu navegador y ve a `http://localhost:8000`
2. **Cargar documentos**: Usa la interfaz para cargar o referenciar materiales del curso
3. **Hacer preguntas**: Escribe tus preguntas en la interfaz de chat
4. **Obtener respuestas**: El sistema busca contenido relevante en la base de datos Qdrant y genera respuestas contextuales usando OpenAI

## 📡 Puntos finales de la API

- `GET /` - Devuelve la interfaz de chat principal
- `POST /chat` - Procesa mensajes de chat y devuelve respuestas generadas por IA

## ⚙️ Variables de entorno

| Variable | Descripción | Por defecto |
|----------|-------------|---------|
| `QDRANT_URL` | URL de la base de datos Qdrant | `http://91.99.108.245:6333` |
| `QDRANT_API_KEY` | Clave de autenticación de la API de Qdrant | `None` |
| `API_KEY_OPENAI` | Clave de API de OpenAI para generación de texto | Requerida |

## 📃 Dependencias

Ver [requirements.txt](requirements.txt) para una lista completa de dependencias incluyendo:
- FastAPI y Uvicorn para el servidor web
- Cliente de Qdrant para operaciones de base de datos vectorial
- Sentence Transformers para incrustaciones
- Cliente de Python de OpenAI
- python-docx para procesamiento de documentos

## 📝 Notas

- Asegúrate de que tu instancia de Qdrant esté correctamente configurada y accesible
- El modelo de incrustaciones (BAAI/bge-base-en-v1.5) se descargará automáticamente en la primera ejecución
- Las llamadas a la API de OpenAI incurrirán en costos según tu uso

## 🐛 Solución de problemas

- **Problemas de conexión**: Verifica la URL de Qdrant y la clave de API en tu archivo `.env`
- **Errores al cargar modelos**: Asegúrate de tener suficiente espacio en disco para el modelo de incrustaciones
- **Errores de OpenAI**: Verifica que tu clave de API sea válida y tenga créditos disponibles

