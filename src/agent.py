import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_port = os.getenv("MYSQL_PORT")
db_name = os.getenv("MYSQL_DATABASE")
ollama_host = os.getenv("OLLAMA_HOST", "ollama-phi3.ia.svc.cluster.local")
ollama_port = os.getenv("OLLAMA_PORT", "11434")

db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

db = SQLDatabase.from_uri(db_url)

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
