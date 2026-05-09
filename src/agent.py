import os
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

# ========== CONFIGURACIÓN ==========
def get_env(name: str, default: str = "") -> str:
    """Obtiene variable de entorno y elimina espacios/saltos de línea."""
    val = os.getenv(name, default)
    return val.strip() if val else ""

db_host = get_env("MYSQL_HOST", "10.110.77.145")   # ClusterIP del servicio
db_port = get_env("MYSQL_PORT", "3306")
db_user = get_env("MYSQL_USER", "invisia")
db_password = get_env("MYSQL_PASSWORD", "Vper1821317@")
db_name = get_env("MYSQL_DATABASE", "invisia_db")

ollama_host = get_env("OLLAMA_HOST", "ollama-phi3.ia.svc.cluster.local")
ollama_port = get_env("OLLAMA_PORT", "11434")

# Validación de variables obligatorias
missing = []
if not db_host: missing.append("MYSQL_HOST")
if not db_user: missing.append("MYSQL_USER")
if not db_password: missing.append("MYSQL_PASSWORD")
if not db_name: missing.append("MYSQL_DATABASE")
if missing:
    raise ValueError(f"Faltan variables de entorno: {', '.join(missing)}")

# ========== CONEXIÓN A MYSQL (ProxySQL) ==========
db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
print(f"Conectando a BD: mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}")

try:
    engine = create_engine(db_url, pool_pre_ping=True)
    # Prueba de conexión simple
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Conexión a BD exitosa.")
except SQLAlchemyError as e:
    print(f"❌ Error conectando a BD: {e}")
    raise

# Crear objeto SQLDatabase para LangChain
db = SQLDatabase(engine)

# ========== CONEXIÓN A OLLAMA ==========
llm = OllamaLLM(
    model="phi3:mini",
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=256
)

# ========== AGENTE TEXT-TO-SQL ==========
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# ========== API ENDPOINTS ==========
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
