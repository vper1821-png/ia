import os
import re
from urllib.parse import quote_plus
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_ollama import OllamaLLM
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

app = FastAPI()

# ── PRUEBA: solo 2 tablas ────────────────────────────────────────────────────
TEST_TABLES = ["apis_logs", "stock_wms"]

# ── Keywords que indican consulta a la BD ────────────────────────────────────
DB_KEYWORDS = [
    "cuanto", "cuantos", "cuanta", "cuantas",
    "stock", "api", "registro", "registros",
    "total", "lista", "dame", "muestra", "muéstrame",
    "hay", "tiene", "tienen", "existe", "existen",
    "ultimo", "último", "primera", "primero",
    "logs", "log", "apis_logs", "stock_wms",
    "consulta", "busca", "encuentra", "filtra",
    "mayor", "menor", "maximo", "minimo", "promedio",
    "contar", "listar", "mostrar", "trae", "traeme",
    "reciente", "recientes", "nuevos", "nuevo",
]


def is_db_question(question: str) -> bool:
    """Determina si la pregunta requiere consultar la BD."""
    q = question.lower().strip()
    return any(kw in q for kw in DB_KEYWORDS)


def clean_env_var(value: str, var_name: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r'[\x00-\x1f\x7f]', '', value).strip()
    if cleaned != value:
        print(f"!! Variable {var_name} fue limpiada")
    return cleaned


# ── Variables ────────────────────────────────────────────────────────────────
db_host     = clean_env_var(os.getenv("MYSQL_HOST",     "10.110.77.145"),                    "MYSQL_HOST")
db_port     = clean_env_var(os.getenv("MYSQL_PORT",     "3306"),                             "MYSQL_PORT")
db_user     = clean_env_var(os.getenv("MYSQL_USER",     "invisia"),                          "MYSQL_USER")
db_password = clean_env_var(os.getenv("MYSQL_PASSWORD", "Vper1821317@"),                     "MYSQL_PASSWORD")
db_name     = clean_env_var(os.getenv("MYSQL_DATABASE", "invisia_db"),                       "MYSQL_DATABASE")
ollama_host = clean_env_var(os.getenv("OLLAMA_HOST",    "ollama-phi3.ia.svc.cluster.local"), "OLLAMA_HOST")
ollama_port = clean_env_var(os.getenv("OLLAMA_PORT",    "11434"),                            "OLLAMA_PORT")

missing = [v for v, val in [
    ("MYSQL_HOST", db_host), ("MYSQL_USER", db_user),
    ("MYSQL_PASSWORD", db_password), ("MYSQL_DATABASE", db_name)
] if not val]
if missing:
    raise ValueError(f"Faltan variables: {', '.join(missing)}")

# ── Conexión BD ──────────────────────────────────────────────────────────────
encoded_password = quote_plus(db_password)
db_url = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
print(f"Conectando a BD: mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}")

try:
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Conexión a BD exitosa.")
except SQLAlchemyError as e:
    print(f"❌ Error conectando a BD: {e}")
    raise

db = SQLDatabase(engine, include_tables=TEST_TABLES)

# Schema fijo — se calcula una vez al arrancar
SCHEMA = db.get_table_info(TEST_TABLES)
print(f"📋 Schema cargado para tablas: {TEST_TABLES}")
print(f"📏 Longitud schema: {len(SCHEMA)} chars")

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = OllamaLLM(
    model="phi3",
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=256,
    stop=["##", "Question:", "Note:", "Explanation:", "\n\n\n"],
)


def extract_sql(raw: str) -> str:
    """Extrae la primera sentencia SQL del output del LLM."""
    raw = re.sub(r'```(?:sql)?', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```', '', raw).strip()
    for line in raw.splitlines():
        line = line.strip()
        if re.match(r'(?i)^(SELECT|WITH)\b', line):
            idx = raw.find(line)
            fragment = raw[idx:]
            if ';' in fragment:
                fragment = fragment[:fragment.index(';') + 1]
            return fragment.strip()
    return raw.split(';')[0].strip() + ';'


def clean_answer(raw: str) -> str:
    """Corta la basura que genera phi3 después de la respuesta."""
    if '\n\n' in raw:
        raw = raw[:raw.index('\n\n')]
    stop_signals = ['##', 'Question:', 'Note:', 'Explanation:', 'Example:', '```']
    for signal in stop_signals:
        if signal in raw:
            raw = raw[:raw.index(signal)]
    return raw.strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────
class Query(BaseModel):
    query: str


@app.post("/generate")
async def generate_sql(query: Query):
    print(f"\n{'='*50}")
    print(f"📥 Pregunta: {query.query}")

    # ── Consulta a la BD ──────────────────────────────────────────────────────
    if is_db_question(query.query):
        print("🗄️ Modo: consulta BD")
        try:
            # Paso 1 — generar SQL
            sql_prompt = (
                f"You are a MySQL expert. Write ONLY a valid MySQL SELECT query. "
                f"No explanations. No markdown. Just SQL.\n\n"
                f"Schema:\n{SCHEMA}\n\n"
                f"Question: {query.query}\n\n"
                f"SQL:"
            )

            print("🤖 Generando SQL...")
            raw_sql = llm.invoke(sql_prompt)
            print(f"🔍 SQL crudo: {raw_sql}")

            sql = extract_sql(raw_sql)
            print(f"✅ SQL: {sql}")

            # Paso 2 — ejecutar SQL
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]

            print(f"📊 Filas: {len(rows)}")

            # Paso 3 — interpretar resultado
            interpret_prompt = (
                f"Answer in one sentence in Spanish. Do not ask questions. Do not add examples.\n\n"
                f"Question: {query.query}\n"
                f"Data: {rows[:10]}\n\n"
                f"Answer:"
            )

            print("🤖 Interpretando...")
            raw_answer = llm.invoke(interpret_prompt)
            answer = clean_answer(raw_answer)
            print(f"💬 Respuesta: {answer}")

            return {"response": answer}

        except SQLAlchemyError as e:
            print(f"❌ SQL error: {e}")
            raise HTTPException(status_code=500, detail=f"Error SQL: {str(e)}")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Conversación general — phi3 responde directamente ────────────────────
    else:
        print("💬 Modo: conversación general")
        try:
            chat_prompt = (
                f"You are a helpful WMS assistant called 'Asistente WMS'. "
                f"Answer naturally in Spanish. Be concise.\n\n"
                f"User: {query.query}\n\n"
                f"Assistant:"
            )
            raw_answer = llm.invoke(chat_prompt)
            answer = clean_answer(raw_answer)
            print(f"💬 Respuesta chat: {answer}")
            return {"response": answer}
        except Exception as e:
            print(f"❌ Error chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "tables": TEST_TABLES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
