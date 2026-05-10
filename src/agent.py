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

# ── Todas las tablas disponibles ─────────────────────────────────────────────
ALL_TABLES = [
    "productos", "stock_wms", "stock_reservas", "clientes", "cliente_direcciones",
    "proveedores", "codigo_barras", "preparacion_kp", "preparacion_archivos",
    "kp_recepcion", "kp_recepcion_archivos", "pedidos_despacho", "detalle_despacho",
    "apis_logs", "apis_config", "inventario_movimientos", "inventario_movimientos_detalle",
    "maestro_clientes", "maestro_tiendas", "empresas", "transportistas", "choferes",
    "exportaciones_historial", "importaciones_historial", "oms_webhook_logs",
]

# ── Mapa de keywords → tablas relevantes ─────────────────────────────────────
TABLE_KEYWORDS = {
    "productos":                  ["producto", "productos", "articulo", "item", "sku", "seleciona", "selecciona"],
    "stock_wms":                  ["stock", "inventario", "disponible", "cantidad", "bodega"],
    "stock_reservas":             ["reserva", "reservas", "reservado"],
    "clientes":                   ["cliente", "clientes", "comprador"],
    "cliente_direcciones":        ["direccion", "direcciones", "domicilio"],
    "proveedores":                 ["proveedor", "proveedores", "supplier"],
    "preparacion_kp":             ["preparacion", "picking", "kp", "preparar"],
    "pedidos_despacho":           ["pedido", "pedidos", "despacho", "despachos", "envio"],
    "detalle_despacho":           ["detalle despacho", "items despacho"],
    "apis_logs":                  ["api", "apis", "log", "logs", "llamada", "request"],
    "apis_config":                ["config api", "configuracion api"],
    "inventario_movimientos":     ["movimiento", "movimientos", "traslado", "entrada", "salida"],
    "kp_recepcion":               ["recepcion", "recepciones", "recibido"],
    "maestro_clientes":           ["falabella", "paris", "retail", "tienda", "maestro cliente"],
    "maestro_tiendas":            ["tienda", "tiendas", "local", "sucursal"],
    "exportaciones_historial":    ["exportacion", "exportaciones", "exportado"],
    "importaciones_historial":    ["importacion", "importaciones", "importado"],
    "transportistas":             ["transportista", "transporte", "carrier"],
    "choferes":                   ["chofer", "choferes", "conductor"],
    "oms_webhook_logs":           ["webhook", "oms"],
}

# ── Keywords que indican consulta a la BD ────────────────────────────────────
DB_KEYWORDS = [
    "cuanto", "cuantos", "cuanta", "cuantas",
    "stock", "api", "registro", "registros",
    "total", "lista", "dame", "muestra", "muestrame",
    "hay", "tiene", "tienen", "existe", "existen",
    "ultimo", "ultimo", "primera", "primero",
    "logs", "log", "apis_logs", "stock_wms",
    "consulta", "busca", "encuentra", "filtra",
    "mayor", "menor", "maximo", "minimo", "promedio",
    "contar", "listar", "mostrar", "trae", "traeme",
    "reciente", "recientes", "nuevos", "nuevo",
    "producto", "productos", "cliente", "clientes",
    "proveedor", "proveedores", "pedido", "pedidos",
    "despacho", "recepcion", "picking", "preparacion",
    "reserva", "movimiento", "exportacion", "importacion",
    "tienda", "tiendas", "transportista",
    "seleciona", "selecciona", "muestra", "listame",
]


def is_db_question(question: str) -> bool:
    q = question.lower().strip()
    return any(kw in q for kw in DB_KEYWORDS)


def get_relevant_tables(question: str) -> list:
    q = question.lower().strip()
    relevant = set()
    for table, keywords in TABLE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            relevant.add(table)
    if not relevant:
        relevant = {"productos", "stock_wms", "clientes", "pedidos_despacho", "apis_logs"}
    return list(relevant)


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
    print("Conexion a BD exitosa.")
except SQLAlchemyError as e:
    print(f"Error conectando a BD: {e}")
    raise

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=256,
    stop=["##", "Question:", "Note:", "Explanation:", "\n\n\n"],
)


def extract_sql(raw: str) -> str:
    raw = re.sub(r'```(?:sql)?', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```', '', raw).strip()
    # Si empieza con SELECT directamente (porque el prompt terminó en SELECT)
    if re.match(r'(?i)^(SELECT|WITH)\b', raw):
        if ';' in raw:
            return raw[:raw.index(';') + 1].strip()
        return raw.strip() + ';'
    for line in raw.splitlines():
        line = line.strip()
        if re.match(r'(?i)^(SELECT|WITH)\b', line):
            idx = raw.find(line)
            fragment = raw[idx:]
            if ';' in fragment:
                fragment = fragment[:fragment.index(';') + 1]
            return fragment.strip()
    return raw.split(';')[0].strip() + ';'


def validate_sql(sql: str) -> tuple:
    if not re.match(r'(?i)^\s*(SELECT|WITH)\b', sql):
        return False, "No es una consulta SELECT valida"
    if re.search(r'(?<![\'"]):\w+', sql):
        return False, "Contiene bind parameters"
    bad_words = ['document', 'write', 'here is', 'this query', 'the following']
    sql_lower = sql.lower()
    for word in bad_words:
        if word in sql_lower[:50]:
            return False, f"Contiene texto no SQL: {word}"
    return True, ""


def get_sql_with_retry(llm_instance, initial_prompt: str, schema: str, question: str, max_retries: int = 3) -> str:
    prompt = initial_prompt
    for attempt in range(max_retries):
        raw_sql = llm_instance.invoke(prompt)
        print(f"SQL crudo (intento {attempt+1}): {raw_sql}")
        # El prompt termina en SELECT, agregar eso al inicio
        sql = "SELECT " + extract_sql(raw_sql) if not re.match(r'(?i)^SELECT', raw_sql.strip()) else extract_sql(raw_sql)
        valid, reason = validate_sql(sql)
        if valid:
            return sql
        print(f"SQL invalido (intento {attempt+1}): {reason}")
        prompt = (
            f"MySQL 8.0 expert. Return ONLY a MySQL SELECT query. No text before or after.\n"
            f"Rules: no bind parameters, no PostgreSQL syntax, use CAST() not ::, use LIMIT 10.\n"
            f"Schema:\n{schema}\n\n"
            f"Question: {question}\n\n"
            f"SELECT"
        )
    raise ValueError(f"No se pudo generar SQL valido despues de {max_retries} intentos")


def clean_answer(raw: str) -> str:
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
    print(f"Pregunta: {query.query}")

    if is_db_question(query.query):
        print("Modo: consulta BD")
        try:
            relevant_tables = get_relevant_tables(query.query)
            print(f"Tablas relevantes: {relevant_tables}")

            db_instance = SQLDatabase(engine, include_tables=relevant_tables)
            schema = db_instance.get_table_info(relevant_tables)

            sql_prompt = (
                "You are a MySQL 8.0 expert. Write ONLY a valid MySQL 8.0 SELECT query.\n"
                "STRICT RULES:\n"
                "- Use only standard MySQL 8.0 syntax\n"
                "- Do NOT use PostgreSQL syntax\n"
                "- Do NOT use :: for casting, use CAST(value AS type) instead\n"
                "- Do NOT use POSITION...FOR syntax\n"
                "- Do NOT use ILIKE, use LIKE instead\n"
                "- Do NOT use bind parameters like :variable\n"
                "- Use LIMIT 10 to avoid returning too many rows\n"
                "- Start response directly with SELECT, no text before or after\n"
                "- No explanations, no markdown, no comments. Just SQL.\n\n"
                f"Schema:\n{schema}\n\n"
                f"Question: {query.query}\n\n"
                "SELECT"
            )

            print("Generando SQL...")
            sql = get_sql_with_retry(llm, sql_prompt, schema, query.query)
            print(f"SQL final: {sql}")

            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]

            print(f"Filas: {len(rows)}")

            interpret_prompt = (
                "Answer in one sentence in Spanish. "
                "Show the ACTUAL values and numbers from the data provided. "
                "Be specific and concrete. Do not be vague. "
                "Do not ask questions. Do not add examples.\n\n"
                f"Question: {query.query}\n"
                f"Data: {rows[:10]}\n\n"
                "Answer:"
            )

            print("Interpretando...")
            raw_answer = llm.invoke(interpret_prompt)
            answer = clean_answer(raw_answer)
            print(f"Respuesta: {answer}")

            return {"response": answer}

        except SQLAlchemyError as e:
            print(f"SQL error: {e}")
            raise HTTPException(status_code=500, detail=f"Error SQL: {str(e)}")
        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    else:
        print("Modo: conversacion general")
        try:
            chat_prompt = (
                "You are a helpful WMS assistant called 'Asistente WMS'. "
                "Answer naturally in Spanish. Be concise.\n\n"
                f"User: {query.query}\n\n"
                "Assistant:"
            )
            raw_answer = llm.invoke(chat_prompt)
            answer = clean_answer(raw_answer)
            print(f"Respuesta chat: {answer}")
            return {"response": answer}
        except Exception as e:
            print(f"Error chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "tables": ALL_TABLES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
