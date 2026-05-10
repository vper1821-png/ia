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

# ------------------------------------------------------------
# 1. Configuración de tablas y keywords (sin cambios)
# ------------------------------------------------------------
ALL_TABLES = [
    "productos", "stock_wms", "stock_reservas", "clientes", "cliente_direcciones",
    "proveedores", "codigo_barras", "preparacion_kp", "preparacion_archivos",
    "kp_recepcion", "kp_recepcion_archivos", "pedidos_despacho", "detalle_despacho",
    "apis_logs", "apis_config", "inventario_movimientos", "inventario_movimientos_detalle",
    "maestro_clientes", "maestro_tiendas", "empresas", "transportistas", "choferes",
    "exportaciones_historial", "importaciones_historial", "oms_webhook_logs",
]

TABLE_KEYWORDS = {
    "productos": ["producto", "productos", "articulo", "item", "sku"],
    "stock_wms": ["stock", "inventario", "disponible", "cantidad", "bodega"],
    "stock_reservas": ["reserva", "reservas", "reservado"],
    "clientes": ["cliente", "clientes", "comprador"],
    "cliente_direcciones": ["direccion", "direcciones", "domicilio"],
    "proveedores": ["proveedor", "proveedores", "supplier"],
    "preparacion_kp": ["preparacion", "picking", "kp", "preparar"],
    "pedidos_despacho": ["pedido", "pedidos", "despacho", "despachos", "envio"],
    "detalle_despacho": ["detalle despacho", "items despacho"],
    "apis_logs": ["api", "apis", "log", "logs", "llamada", "request"],
    "apis_config": ["config api", "configuracion api"],
    "inventario_movimientos": ["movimiento", "movimientos", "traslado", "entrada", "salida"],
    "kp_recepcion": ["recepcion", "recepciones", "recibido"],
    "maestro_clientes": ["falabella", "paris", "retail", "tienda", "maestro cliente"],
    "maestro_tiendas": ["tienda", "tiendas", "local", "sucursal"],
    "exportaciones_historial": ["exportacion", "exportaciones", "exportado"],
    "importaciones_historial": ["importacion", "importaciones", "importado"],
    "transportistas": ["transportista", "transporte", "carrier"],
    "choferes": ["chofer", "choferes", "conductor"],
    "oms_webhook_logs": ["webhook", "oms"],
}

DB_KEYWORDS = [
    "cuanto", "cuantos", "cuanta", "cuantas", "stock", "api", "registro", "registros",
    "total", "lista", "dame", "muestra", "muestrame", "hay", "tiene", "tienen",
    "existe", "existen", "ultimo", "primera", "primero", "logs", "log", "stock_wms",
    "consulta", "busca", "encuentra", "filtra", "mayor", "menor", "maximo", "minimo",
    "promedio", "contar", "listar", "mostrar", "trae", "traeme", "reciente", "recientes",
    "nuevos", "nuevo", "producto", "productos", "cliente", "clientes", "proveedor",
    "proveedores", "pedido", "pedidos", "despacho", "recepcion", "picking", "preparacion",
    "reserva", "movimiento", "exportacion", "importacion", "tienda", "tiendas", "transportista",
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

# ------------------------------------------------------------
# 2. Variables de entorno y conexión a BD
# ------------------------------------------------------------
db_host     = clean_env_var(os.getenv("MYSQL_HOST",     "10.110.77.145"), "MYSQL_HOST")
db_port     = clean_env_var(os.getenv("MYSQL_PORT",     "3306"), "MYSQL_PORT")
db_user     = clean_env_var(os.getenv("MYSQL_USER",     "invisia"), "MYSQL_USER")
db_password = clean_env_var(os.getenv("MYSQL_PASSWORD", "Vper1821317@"), "MYSQL_PASSWORD")
db_name     = clean_env_var(os.getenv("MYSQL_DATABASE", "invisia_db"), "MYSQL_DATABASE")
ollama_host = clean_env_var(os.getenv("OLLAMA_HOST",    "ollama-phi3.ia.svc.cluster.local"), "OLLAMA_HOST")
ollama_port = clean_env_var(os.getenv("OLLAMA_PORT",    "11434"), "OLLAMA_PORT")

missing = [v for v, val in [
    ("MYSQL_HOST", db_host), ("MYSQL_USER", db_user),
    ("MYSQL_PASSWORD", db_password), ("MYSQL_DATABASE", db_name)
] if not val]
if missing:
    raise ValueError(f"Faltan variables: {', '.join(missing)}")

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

# ------------------------------------------------------------
# 3. Configuración del LLM (más permisiva y determinista)
# ------------------------------------------------------------
llm = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=512,
    top_k=1,
    stop=None,   # quitamos stops para no truncar
)

# ------------------------------------------------------------
# 4. Funciones de extracción y validación de SQL (mejoradas)
# ------------------------------------------------------------
def extract_sql(raw: str) -> str:
    """Extrae la PRIMERA sentencia SELECT válida, ignorando texto alrededor."""
    raw = raw.strip()
    # Eliminar bloques markdown
    raw = re.sub(r'```(?:sql)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```', '', raw).strip()
    # Buscar SELECT ... hasta ; o fin de línea
    match = re.search(r'(SELECT\s+.*?)(;|\n\s*$)', raw, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        if not sql.endswith(';'):
            sql += ';'
        # Corregir duplicado "SELECT SELECT"
        if sql.upper().startswith('SELECT SELECT'):
            sql = sql[7:]  # elimina el primer SELECT
        return sql
    # Si no encuentra, buscar cualquier línea que empiece por SELECT
    for line in raw.splitlines():
        if re.match(r'(?i)^SELECT\b', line.strip()):
            sql = line.strip()
            if not sql.endswith(';'):
                sql += ';'
            return sql
    return ""

def validate_sql(sql: str) -> bool:
    """Valida que sea una consulta SELECT (sin restricciones excesivas)."""
    return bool(re.match(r'(?i)^\s*SELECT\b', sql.strip()))

def fallback_sql(question: str) -> str:
    """Genera una consulta SQL por defecto según la pregunta."""
    q = question.lower()
    if 'producto' in q:
        if 'cuantos' in q or 'total' in q:
            return "SELECT COUNT(*) FROM productos;"
        return "SELECT * FROM productos LIMIT 10;"
    if 'cliente' in q:
        if 'cuantos' in q or 'total' in q:
            return "SELECT COUNT(*) FROM clientes;"
        return "SELECT * FROM clientes LIMIT 10;"
    if 'proveedor' in q:
        return "SELECT * FROM proveedores LIMIT 10;"
    if 'chofer' in q:
        return "SELECT * FROM choferes LIMIT 10;"
    if 'transportista' in q:
        return "SELECT * FROM transportistas LIMIT 10;"
    if 'stock' in q:
        return "SELECT * FROM stock_wms LIMIT 10;"
    if 'pedido' in q or 'despacho' in q:
        return "SELECT * FROM pedidos_despacho LIMIT 10;"
    if 'log' in q or 'api' in q:
        return "SELECT * FROM apis_logs ORDER BY fecha DESC LIMIT 10;"
    # Si nada coincide, devuelve un mensaje amigable como consulta
    return "SELECT 'No se pudo generar la consulta automática' AS mensaje;"

def get_sql_with_retry(llm_instance, schema: str, question: str, max_retries: int = 2) -> str:
    """Intenta obtener SQL del LLM, reintentando si falla."""
    # Prompt muy corto y directo, con ejemplos incrustados
    base_prompt = f"""Eres un generador de SQL para MySQL. Responde ÚNICAMENTE con la sentencia SQL SELECT que resuelva la pregunta.
No añadas texto, explicaciones, comillas, markdown ni nada más.
Ejemplos:
Pregunta: "cuantos productos tengo"
SQL: SELECT COUNT(*) FROM productos;

Pregunta: "muestra los primeros 5 clientes"
SQL: SELECT * FROM clientes LIMIT 5;

Esquema de tablas relevantes:
{schema}

Pregunta: {question}
SQL:"""

    for attempt in range(max_retries):
        raw = llm_instance.invoke(base_prompt)
        print(f"SQL crudo (intento {attempt+1}): {raw[:300]}")
        sql = extract_sql(raw)
        if sql and validate_sql(sql):
            return sql
        print(f"Intento {attempt+1} fallido. Reintentando...")
        # Mensaje correctivo
        base_prompt = f"La respuesta anterior no fue una consulta SQL válida. Por favor, escribe solo la sentencia SELECT para: {question}\nSQL:"
    
    print("Usando fallback después de reintentos fallidos.")
    return fallback_sql(question)

def clean_answer(raw: str) -> str:
    """Limpia la respuesta final para que sea una frase corta."""
    raw = raw.strip()
    # Cortar en doble salto de línea o palabras clave
    if '\n\n' in raw:
        raw = raw[:raw.index('\n\n')]
    stop_signals = ['##', 'Question:', 'Note:', 'Explanation:', 'Example:', '```']
    for signal in stop_signals:
        if signal in raw:
            raw = raw[:raw.index(signal)]
    return raw.strip()

# ------------------------------------------------------------
# 5. Endpoint principal
# ------------------------------------------------------------
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

            print("Generando SQL...")
            sql = get_sql_with_retry(llm, schema, query.query)
            print(f"SQL final: {sql}")

            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]

            print(f"Filas obtenidas: {len(rows)}")

            # Interpretación de resultados
            interpret_prompt = (
                "Responde en una sola oración en español. "
                "Muestra los valores y números reales de los datos proporcionados. "
                "Sé específico. No seas vago.\n\n"
                f"Pregunta: {query.query}\n"
                f"Datos: {rows[:10]}\n\n"
                "Respuesta:"
            )
            raw_answer = llm.invoke(interpret_prompt)
            answer = clean_answer(raw_answer)
            print(f"Respuesta: {answer}")
            return {"response": answer}

        except SQLAlchemyError as e:
            print(f"Error SQL: {e}")
            raise HTTPException(status_code=500, detail=f"Error al ejecutar consulta: {str(e)}")
        except Exception as e:
            print(f"Error inesperado: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    else:
        print("Modo: conversación general")
        try:
            chat_prompt = (
                "Eres un asistente útil del sistema WMS llamado 'Asistente WMS'. "
                "Responde de forma natural en español, de manera concisa.\n\n"
                f"Usuario: {query.query}\n\n"
                "Asistente:"
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
