import os
import re
from urllib.parse import quote_plus
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
    """Elimina solo espacios, tabuladores y saltos de línea al inicio y final.
    No elimina caracteres especiales (como @, #, etc.) importantes para contraseñas."""
    if not value:
        return ""
    # Eliminar caracteres de control (excepto caracteres imprimibles)
    cleaned = re.sub(r'[\x00-\x1f\x7f]', '', value)  # quita ASCII control
    cleaned = cleaned.strip()
    if cleaned != value:
        print(f"!! Variable {var_name} fue limpiada (caracteres de control eliminados)")
    return cleaned

# Obtener variables
db_host = clean_env_var(os.getenv("MYSQL_HOST", "10.110.77.145"), "MYSQL_HOST")
db_port = clean_env_var(os.getenv("MYSQL_PORT", "3306"), "MYSQL_PORT")
db_user = clean_env_var(os.getenv("MYSQL_USER", "invisia"), "MYSQL_USER")
db_password_raw = os.getenv("MYSQL_PASSWORD", "Vper1821317@")
db_password = clean_env_var(db_password_raw, "MYSQL_PASSWORD")
db_name = clean_env_var(os.getenv("MYSQL_DATABASE", "invisia_db"), "MYSQL_DATABASE")
ollama_host = clean_env_var(os.getenv("OLLAMA_HOST", "ollama-phi3.ia.svc.cluster.local"), "OLLAMA_HOST")
ollama_port = clean_env_var(os.getenv("OLLAMA_PORT", "11434"), "OLLAMA_PORT")

# Validación
missing = []
if not db_host: missing.append("MYSQL_HOST")
if not db_user: missing.append("MYSQL_USER")
if not db_password: missing.append("MYSQL_PASSWORD")
if not db_name: missing.append("MYSQL_DATABASE")
if missing:
    raise ValueError(f"Faltan variables de entorno: {', '.join(missing)}")

# Codificar la contraseña para URL
encoded_password = quote_plus(db_password)

# Construir URL con contraseña codificada
db_url = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
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

# Conexión a Ollama (igual)
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
