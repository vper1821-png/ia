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
    # Ignorar prompts internos de Open WebUI (títulos, tags, seguimiento)
    internal_prompts = [
        "### task:", "suggest 3-5 relevant follow-up", "generate a concise",
        "generate 1-3 broad tags", "follow_ups", "json format",
        "title with an emoji", "chat history"
    ]
    if any(p in q for p in internal_prompts):
        return False
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
# 3. Configuración del LLM (más capacidad para listados)
# ------------------------------------------------------------
llm = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=1024,   # aumentado para respuestas largas
    top_k=1,
    stop=None,
)

# ------------------------------------------------------------
# 4. Funciones de extracción y validación de SQL
# ------------------------------------------------------------
def extract_sql(raw: str) -> str:
    """Extrae la PRIMERA sentencia SELECT válida."""
    raw = raw.strip()
    raw = re.sub(r'```(?:sql)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```', '', raw).strip()
    match = re.search(r'(SELECT\s+.*?)(;|\n\s*$)', raw, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        if not sql.endswith(';'):
            sql += ';'
        if sql.upper().startswith('SELECT SELECT'):
            sql = sql[7:]
        return sql
    for line in raw.splitlines():
        if re.match(r'(?i)^SELECT\b', line.strip()):
            sql = line.strip()
            if not sql.endswith(';'):
                sql += ';'
            return sql
    return ""

def validate_sql(sql: str) -> bool:
    return bool(re.match(r'(?i)^\s*SELECT\b', sql.strip()))

def fallback_sql(question: str) -> str:
    q = question.lower()
    if 'producto' in q:
        return "SELECT * FROM productos ORDER BY id LIMIT 10;" if 'list' in q or 'muestra' in q else "SELECT COUNT(*) FROM productos;"
    if 'cliente' in q:
        return "SELECT * FROM clientes ORDER BY id LIMIT 10;" if 'list' in q or 'muestra' in q else "SELECT COUNT(*) FROM clientes;"
    if 'proveedor' in q:
        return "SELECT * FROM proveedores ORDER BY id LIMIT 10;"
    if 'chofer' in q:
        return "SELECT * FROM choferes ORDER BY id LIMIT 10;"
    if 'transportista' in q:
        return "SELECT * FROM transportistas ORDER BY id LIMIT 10;"
    if 'stock' in q:
        return "SELECT * FROM stock_wms ORDER BY id LIMIT 10;"
    if 'pedido' in q or 'despacho' in q:
        return "SELECT * FROM pedidos_despacho ORDER BY fecha DESC LIMIT 10;"
    if 'log' in q or 'api' in q:
        return "SELECT * FROM apis_logs ORDER BY fecha DESC LIMIT 10;"
    return "SELECT 'No se pudo generar la consulta automática' AS mensaje;"

def get_sql_with_retry(llm_instance, schema: str, question: str, max_retries: int = 2) -> str:
    base_prompt = f"""Eres un generador de SQL para MySQL. Responde ÚNICAMENTE con la sentencia SQL SELECT que resuelva la pregunta.
No añadas texto, explicaciones, comillas, markdown ni nada más.
Si la pregunta pide un listado, incluye ORDER BY por la columna más lógica (ej. nombre, id o fecha). Usa LIMIT 10 si no se especifica la cantidad.
Ejemplos:
Pregunta: "cuantos productos tengo"
SQL: SELECT COUNT(*) FROM productos;

Pregunta: "muestra los primeros 5 clientes"
SQL: SELECT * FROM clientes ORDER BY id LIMIT 5;

Pregunta: "listame los productos ordenados por nombre"
SQL: SELECT * FROM productos ORDER BY nombre LIMIT 10;

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
        base_prompt = f"La respuesta anterior no fue una consulta SQL válida. Por favor, escribe solo la sentencia SELECT para: {question}\nSQL:"
    
    print("Usando fallback después de reintentos fallidos.")
    return fallback_sql(question)

# ------------------------------------------------------------
# 5. Nueva función para formatear y analizar resultados
# ------------------------------------------------------------
def format_and_analyze_results(question: str, rows: list) -> str:
    total = len(rows)
    if total == 0:
        return "No se encontraron resultados para tu consulta."
    
    # Tomar una muestra (máximo 10 filas) para el análisis
    sample = rows[:10]
    # Construir un texto legible con la muestra
    sample_text = ""
    for i, row in enumerate(sample, 1):
        # Extraer los valores más relevantes: si hay 'nombre' o 'descripcion' o la primera columna
        if 'nombre_cliente' in row:
            name = row.get('nombre_cliente', 'N/A')
            rut = row.get('rut_cliente', '')
            sample_text += f"{i}. {name} (RUT: {rut})\n"
        elif 'nombre' in row:
            name = row.get('nombre', 'N/A')
            sample_text += f"{i}. {name}\n"
        else:
            # Mostrar el primer valor de la fila
            first_val = list(row.values())[0] if row else ''
            sample_text += f"{i}. {first_val}\n"
    
    prompt = f"""Eres un asistente de análisis de datos. A continuación tienes el resultado de una consulta a una base de datos.
Pregunta original: {question}
Número total de registros: {total}
Muestra de los primeros {len(sample)} registros:
{sample_text}

Responde en español de forma clara y útil. Haz lo siguiente:
1. Indica el número total de registros encontrados.
2. Si es un listado, muestra los primeros (o últimos) registros de forma ordenada y legible, usando el formato de lista numerada.
3. Proporciona un breve análisis: ¿qué se puede concluir de estos datos? (por ejemplo, el valor máximo, mínimo, tendencia, o simplemente que son los más recientes).
4. Sé conciso pero informativo. No uses lenguaje vago.
5. Si la pregunta pedía un listado, asegúrate de que la respuesta incluya los elementos concretos.

Respuesta:"""
    
    try:
        analysis = llm.invoke(prompt)
        return analysis.strip()
    except Exception as e:
        # Fallback: respuesta manual
        return f"Se encontraron {total} registros. Los primeros {len(sample)} son:\n{sample_text}"

# ------------------------------------------------------------
# 6. Endpoint principal mejorado
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

            # Usar la nueva función de análisis
            answer = format_and_analyze_results(query.query, rows)
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
            # Limpieza básica
            answer = re.sub(r'```.*?```', '', raw_answer, flags=re.DOTALL).strip()
            if '\n\n' in answer:
                answer = answer[:answer.index('\n\n')]
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
