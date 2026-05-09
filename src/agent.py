import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Leer y limpiar variables de entorno
def get_env(name: str, default: str = ""):
    val = os.getenv(name, default)
    if val:
        val = val.strip()
    return val

db_user = (os.getenv("MYSQL_USER") or "").strip()
db_password = (os.getenv("MYSQL_PASSWORD") or "").strip()
db_host = (os.getenv("MYSQL_HOST") or "").strip()
db_port = (os.getenv("MYSQL_PORT") or "3306").strip()
db_name = (os.getenv("MYSQL_DATABASE") or "").strip()

print(f"DEBUG: MYSQL_USER='{db_user}' (len={len(db_user)})")
print(f"DEBUG: MYSQL_HOST='{db_host}'")
print(f"DEBUG: MYSQL_DATABASE='{db_name}'")
if not db_user:
    print("ERROR: MYSQL_USER está vacío")
    raise ValueError("MYSQL_USER no puede estar vacío")
ollama_host = get_env("OLLAMA_HOST", "ollama-phi3.ia.svc.cluster.local")
ollama_port = get_env("OLLAMA_PORT", "11434")

# Validar que todas las variables de BD estén presentes
missing = []
if not db_user: missing.append("MYSQL_USER")
if not db_password: missing.append("MYSQL_PASSWORD")
if not db_host: missing.append("MYSQL_HOST")
if not db_name: missing.append("MYSQL_DATABASE")
if missing:
    error_msg = f"Faltan variables de entorno: {', '.join(missing)}"
    print(error_msg)
    raise ValueError(error_msg)

# Construir URL de conexión
db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# Mostrar URL ocultando la contraseña para depuración
masked_url = f"mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}"
print(f"Conectando a BD: {masked_url}")

# Conectar a la base de datos
try:
    db = SQLDatabase.from_uri(db_url)
    print("Conexión a BD exitosa.")
except Exception as e:
    print(f"Error conectando a BD: {e}")
    raise

# Conectar a Ollama
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
