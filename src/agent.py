import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_ollama import OllamaLLM
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

def clean_env_var(value: str, var_name: str) -> str:
    """Limpia caracteres no imprimibles y espacios alrededor."""
    if not value:
        return ""
    # Eliminar cualquier carácter que no sea letra, dígito, punto, dos puntos o guión
    cleaned = re.sub(r'[^\w\.:-]', '', value)
    # También eliminar posibles retornos de carro y saltos de línea
    cleaned = cleaned.replace('\r', '').replace('\n', '').strip()
    if cleaned != value:
        print(f"!! Variable {var_name} fue limpiada: original='{value}' -> limpia='{cleaned}'")
    return cleaned

# Obtener variables
db_host_raw = os.getenv("MYSQL_HOST", "10.110.77.145")
db_host = clean_env_var(db_host_raw, "MYSQL_HOST")

db_port_raw = os.getenv("MYSQL_PORT", "3306")
db_port = clean_env_var(db_port_raw, "MYSQL_PORT")

db_user_raw = os.getenv("MYSQL_USER", "invisia")
db_user = clean_env_var(db_user_raw, "MYSQL_USER")

db_password_raw = os.getenv("MYSQL_PASSWORD", "Vper1821317@")
db_password = clean_env_var(db_password_raw, "MYSQL_PASSWORD")

db_name_raw = os.getenv("MYSQL_DATABASE", "invisia_db")
db_name = clean_env_var(db_name_raw, "MYSQL_DATABASE")

ollama_host_raw = os.getenv("OLLAMA_HOST", "ollama-phi3.ia.svc.cluster.local")
ollama_host = clean_env_var(ollama_host_raw, "OLLAMA_HOST")
ollama_port_raw = os.getenv("OLLAMA_PORT", "11434")
ollama_port = clean_env_var(ollama_port_raw, "OLLAMA_PORT")

# Validación
missing = []
if not db_host: missing.append("MYSQL_HOST")
if not db_port: missing.append("MYSQL_PORT")
if not db_user: missing.append("MYSQL_USER")
if not db_password: missing.append("MYSQL_PASSWORD")
if not db_name: missing.append("MYSQL_DATABASE")
if missing:
    raise ValueError(f"Faltan variables de entorno: {', '.join(missing)}")

print(f"DEBUG: MYSQL_HOST='{db_host}' (len={len(db_host)})")
print(f"DEBUG: MYSQL_HOST raw chars: {[hex(ord(c)) for c in db_host]}")

# Construir URL usando nombre DNS (más robusto)
# Si se prefiere IP, usar db_host; pero intentemos antes con el nombre DNS del servicio de Kubernetes
dns_host = "invisia-pxc-proxysql.database.svc.cluster.local"
db_url = f"mysql+pymysql://{db_user}:{db_password}@{dns_host}:{db_port}/{db_name}"
print(f"Conectando a BD (DNS): mysql+pymysql://{db_user}:***@{dns_host}:{db_port}/{db_name}")

try:
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Conexión a BD exitosa (usando DNS).")
except Exception as e1:
    print(f"❌ Falló conexión con DNS: {e1}")
    # Segunda opción: usar la IP limpia
    db_url_ip = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    print(f"Conectando a BD (IP): mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}")
    try:
        engine = create_engine(db_url_ip, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Conexión a BD exitosa (usando IP).")
    except Exception as e2:
        print(f"❌ Error definitivo conectando a BD: {e2}")
        raise

db = SQLDatabase(engine)

# Ollama y agente (sin cambios relevantes)
llm = OllamaLLM(
    model="phi3:mini",
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=256
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

class Query(BaseModel):
    query: str

@app.post("/generate")
async def generate_sql(query: Query):
    try:
        response = agent.invoke({"input": query.query})
        return {"response": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
