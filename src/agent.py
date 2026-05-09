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
 
load_dotenv()
 
app = FastAPI()
 
 
def clean_env_var(value: str, var_name: str) -> str:
    """Elimina solo espacios, tabuladores y saltos de línea al inicio y final.
    No elimina caracteres especiales (como @, #, etc.) importantes para contraseñas."""
    if not value:
        return ""
    cleaned = re.sub(r'[\x00-\x1f\x7f]', '', value)
    cleaned = cleaned.strip()
    if cleaned != value:
        print(f"!! Variable {var_name} fue limpiada (caracteres de control eliminados)")
    return cleaned
 
 
# Obtener variables
db_host     = clean_env_var(os.getenv("MYSQL_HOST",     "10.110.77.145"),                    "MYSQL_HOST")
db_port     = clean_env_var(os.getenv("MYSQL_PORT",     "3306"),                             "MYSQL_PORT")
db_user     = clean_env_var(os.getenv("MYSQL_USER",     "invisia"),                          "MYSQL_USER")
db_password = clean_env_var(os.getenv("MYSQL_PASSWORD", "Vper1821317@"),                     "MYSQL_PASSWORD")
db_name     = clean_env_var(os.getenv("MYSQL_DATABASE", "invisia_db"),                       "MYSQL_DATABASE")
ollama_host = clean_env_var(os.getenv("OLLAMA_HOST",    "ollama-phi3.ia.svc.cluster.local"), "OLLAMA_HOST")
ollama_port = clean_env_var(os.getenv("OLLAMA_PORT",    "11434"),                            "OLLAMA_PORT")
 
# Validación
missing = []
if not db_host:     missing.append("MYSQL_HOST")
if not db_user:     missing.append("MYSQL_USER")
if not db_password: missing.append("MYSQL_PASSWORD")
if not db_name:     missing.append("MYSQL_DATABASE")
if missing:
    raise ValueError(f"Faltan variables de entorno: {', '.join(missing)}")
 
# Conexión a BD
encoded_password = quote_plus(db_password)
db_url    = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
masked_url = f"mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}"
print(f"Conectando a BD: {masked_url}")
 
try:
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Conexión a BD exitosa.")
except SQLAlchemyError as e:
    print(f"❌ Error definitivo conectando a BD: {e}")
    raise
 
db = SQLDatabase(engine)
 
# Conexión a Ollama
llm = OllamaLLM(
    model="phi3:mini",
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=512,
)
 
 
def extract_sql(raw: str) -> str:
    """Extrae la primera sentencia SQL limpia del output del LLM."""
    # Quitar bloques markdown ```sql ... ``` o ``` ... ```
    raw = re.sub(r'```(?:sql)?', '', raw, flags=re.IGNORECASE).strip()
 
    # Quedarse solo con la primera línea que empiece con SELECT/INSERT/UPDATE/DELETE
    for line in raw.splitlines():
        line = line.strip()
        if re.match(r'(?i)^(SELECT|INSERT|UPDATE|DELETE|WITH)\b', line):
            # Tomar desde esa línea hasta el primer ; o hasta el final
            idx = raw.find(line)
            fragment = raw[idx:]
            # Cortar en el primer punto y coma
            if ';' in fragment:
                fragment = fragment[:fragment.index(';') + 1]
            return fragment.strip()
 
    # Fallback: devolver lo que haya limpio
    return raw.split(';')[0].strip() + ';'
 
 
class Query(BaseModel):
    query: str
 
 
@app.post("/generate")
async def generate_sql(query: Query):
    try:
        # ── Paso 1: obtener schema ──────────────────────────────────────────
        tables = db.get_usable_table_names()
        # Limitar a 15 tablas para no saturar el contexto de phi3:mini
        schema = db.get_table_info(tables[:15])
 
        print(f"📥 Pregunta recibida: {query.query}")
        print(f"📋 Tablas disponibles: {tables}")
 
        # ── Paso 2: generar SQL ─────────────────────────────────────────────
        sql_prompt = f"""You are a MySQL expert. Given the database schema below, write ONLY the SQL query that answers the question. Do not explain. Do not add comments. Output only the SQL statement.
 
Schema:
{schema}
 
Question: {query.query}
 
SQL:"""
 
        print("🤖 Llamando a phi3 para generar SQL...")
        raw_sql = llm.invoke(sql_prompt)
        print(f"🔍 SQL crudo: {raw_sql}")
 
        sql = extract_sql(raw_sql)
        print(f"✅ SQL limpio: {sql}")
 
        # ── Paso 3: ejecutar SQL ────────────────────────────────────────────
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(r._mapping) for r in result]
 
        print(f"📊 Filas obtenidas: {len(rows)}")
 
        # Truncar si hay muchas filas para no saturar el contexto
        rows_for_llm = rows[:20]
 
        # ── Paso 4: interpretar resultado ───────────────────────────────────
        interpret_prompt = f"""You are a helpful assistant. Answer the following question in Spanish based on the query result. Be concise and clear.
 
Question: {query.query}
SQL executed: {sql}
Result: {rows_for_llm}
 
Answer in Spanish:"""
 
        print("🤖 Llamando a phi3 para interpretar resultado...")
        final_answer = llm.invoke(interpret_prompt)
        print(f"💬 Respuesta final: {final_answer}")
 
        return {"response": final_answer.strip()}
 
    except SQLAlchemyError as e:
        print(f"❌ Error SQL: {e}")
        raise HTTPException(status_code=500, detail=f"Error ejecutando SQL: {str(e)}")
    except Exception as e:
        print(f"❌ Error general: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/health")
async def health():
    return {"status": "ok"}
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
