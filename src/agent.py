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

# ── Mapa semántico: keywords → tablas relevantes ────────────────────────────
TABLE_KEYWORDS = {
    "pedido":           ["pedidos_despacho", "preparacion_kp", "detalle_despacho"],
    "despacho":         ["pedidos_despacho", "detalle_despacho", "manifiesto", "manifiesto_detalle"],
    "orden":            ["pedidos_despacho", "oc", "detalle_oc"],
    "recepcion":        ["recepciones", "recepcion_detalle", "kp_recepcion", "kp_recepcion_archivos"],
    "stock":            ["stock", "stock_wms", "stock_reservas", "inventario_movimientos", "inventario_movimientos_detalle"],
    "producto":         ["productos", "codigo_barras", "catalogos_productos"],
    "cliente":          ["clientes", "cliente_direcciones", "maestro_clientes"],
    "proveedor":        ["proveedores", "catalogos_proveedores"],
    "tienda":           ["maestro_tiendas"],
    "ubicacion":        ["ubicaciones"],
    "movimiento":       ["inventario_movimientos", "inventario_movimientos_detalle", "movimientos_stock"],
    "preparacion":      ["preparacion_kp", "preparacion_archivos"],
    "manifiesto":       ["manifiesto", "manifiesto_detalle"],
    "traslado":         ["traslados_internos"],
    "empresa":          ["empresas"],
    "transportista":    ["transportistas", "vehiculos", "choferes"],
    "oc":               ["oc", "detalle_oc", "kp_oc"],
    "compra":           ["oc", "detalle_oc"],
    "usuario":          ["login", "perfiles", "perfil_menu"],
    "api":              ["apis_config", "api_users", "api_jwt_tokens"],
    "exportacion":      ["exportaciones_config", "exportaciones_historial"],
    "importacion":      ["importacion_formatos", "importaciones_historial"],
    "kp":               ["kp_recepcion", "kp_oc", "kp_despacho_ln"],
    "bloqueo":          ["bloqueos_horarios", "horarios_atencion"],
    "muelle":           ["muelles"],
    "sap":              ["sap_integration_log"],
}

ALL_TABLES = [
    "api_jwt_tokens", "api_users", "apis_config", "apis_headers", "apis_logs",
    "apis_params", "bloqueos_horarios", "catalogos_productos", "catalogos_proveedores",
    "choferes", "citas", "cliente_direcciones", "clientes", "codigo_barras",
    "configuraciones_tiempo", "detalle_despacho", "detalle_oc", "empresas",
    "exportaciones_config", "exportaciones_historial", "historial_estados",
    "horarios_atencion", "importacion_formatos", "importaciones_historial",
    "importaciones_oc_historial", "integraciones_log", "inventario_movimientos",
    "inventario_movimientos_detalle", "kp_despacho_ln", "kp_despacho_ln_archivos",
    "kp_oc", "kp_oc_archivos", "kp_recepcion", "kp_recepcion_archivos", "login",
    "maestro_clientes", "maestro_tiendas", "manifiesto", "manifiesto_detalle",
    "menu", "movimientos_stock", "muelles", "necesidad_archivos", "necesidad_pedido",
    "oc", "oms_webhook_logs", "pedidos_despacho", "perfil_menu", "perfiles",
    "preparacion_archivos", "preparacion_kp", "productos", "proveedores",
    "recepcion_detalle", "recepciones", "sap_integration_log", "stock",
    "stock_reservas", "stock_reservas_archivos", "stock_wms", "stock_wms_archivos",
    "transportistas", "traslados_internos", "ubicaciones",
]


def clean_env_var(value: str, var_name: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r'[\x00-\x1f\x7f]', '', value).strip()
    if cleaned != value:
        print(f"!! Variable {var_name} fue limpiada")
    return cleaned


# ── Variables de entorno ─────────────────────────────────────────────────────
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
    raise ValueError(f"Faltan variables de entorno: {', '.join(missing)}")

# ── Conexión BD ──────────────────────────────────────────────────────────────
encoded_password = quote_plus(db_password)
db_url     = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
masked_url = f"mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}"
print(f"Conectando a BD: {masked_url}")

try:
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Conexión a BD exitosa.")
except SQLAlchemyError as e:
    print(f"❌ Error conectando a BD: {e}")
    raise

db = SQLDatabase(engine)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = OllamaLLM(
    model="phi3:mini",
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=512,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def select_tables(question: str) -> list:
    """Selecciona tablas relevantes según keywords en la pregunta."""
    q = question.lower()
    selected = set()
    for keyword, tables in TABLE_KEYWORDS.items():
        if keyword in q:
            selected.update(tables)
    # Fallback: tablas operativas principales si no matchea nada
    if not selected:
        selected = {
            "pedidos_despacho", "preparacion_kp", "detalle_despacho",
            "stock_wms", "clientes", "productos",
        }
    valid = [t for t in selected if t in ALL_TABLES]
    print(f"📋 Tablas seleccionadas: {valid}")
    return valid


def extract_sql(raw: str) -> str:
    """Extrae la primera sentencia SQL limpia del output del LLM."""
    raw = re.sub(r'```(?:sql)?', '', raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r'```', '', raw).strip()

    for line in raw.splitlines():
        line = line.strip()
        if re.match(r'(?i)^(SELECT|INSERT|UPDATE|DELETE|WITH)\b', line):
            idx = raw.find(line)
            fragment = raw[idx:]
            if ';' in fragment:
                fragment = fragment[:fragment.index(';') + 1]
            return fragment.strip()

    return raw.split(';')[0].strip() + ';'


# ── Endpoints ─────────────────────────────────────────────────────────────────
class Query(BaseModel):
    query: str


@app.post("/generate")
async def generate_sql(query: Query):
    try:
        print(f"\n{'='*60}")
        print(f"📥 Pregunta: {query.query}")

        # Paso 1: seleccionar tablas relevantes y obtener schema
        tables = select_tables(query.query)
        schema = db.get_table_info(tables)

        # Paso 2: generar SQL con phi3
        sql_prompt = f"""You are a MySQL expert working with a WMS (Warehouse Management System) database.
Given the schema below, write ONLY a valid MySQL SELECT query that answers the question.
Rules:
- Output ONLY the SQL statement, nothing else.
- Do not add explanations, comments or markdown.
- Use LIMIT 20 unless the question asks for totals or counts.
- Use table aliases for clarity.

Schema:
{schema}

Question: {query.query}

SQL:"""

        print("🤖 Generando SQL...")
        raw_sql = llm.invoke(sql_prompt)
        print(f"🔍 SQL crudo:\n{raw_sql}")

        sql = extract_sql(raw_sql)
        print(f"✅ SQL limpio: {sql}")

        # Paso 3: ejecutar SQL
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(r._mapping) for r in result]

        print(f"📊 Filas obtenidas: {len(rows)}")
        rows_for_llm = rows[:20]

        # Paso 4: interpretar resultado en español
        interpret_prompt = f"""You are a helpful WMS assistant. Answer the question in Spanish based on the data below. Be concise and clear. Do not mention SQL.

Question: {query.query}
Data: {rows_for_llm}

Answer in Spanish:"""

        print("🤖 Interpretando resultado...")
        final_answer = llm.invoke(interpret_prompt)
        final_answer = final_answer.strip()
        print(f"💬 Respuesta: {final_answer}")

        return {"response": final_answer}

    except SQLAlchemyError as e:
        msg = f"Error ejecutando SQL: {str(e)}"
        print(f"❌ {msg}")
        raise HTTPException(status_code=500, detail=msg)
    except Exception as e:
        msg = str(e)
        print(f"❌ Error general: {msg}")
        raise HTTPException(status_code=500, detail=msg)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
