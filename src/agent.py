"""
WMS LLM Assistant — Versión mejorada con esquema completo y contexto enriquecido.
"""
import os
import re
import json
import logging
from urllib.parse import quote_plus
from datetime import datetime

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wms_llm")

app = FastAPI(title="WMS LLM Assistant", version="2.0")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CATÁLOGO COMPLETO DE TABLAS CON DESCRIPCIÓN Y COLUMNAS CLAVE
# ─────────────────────────────────────────────────────────────────────────────

# Mapa: nombre_tabla → (descripción, columnas_clave, keywords)
TABLE_CATALOG = {
    # Maestros de datos
    "productos": {
        "desc": "Catálogo maestro de productos/SKUs del WMS. Contiene código SAP, descripción, familia, subfamilia, talla, color, procedencia, etc.",
        "cols": "id, empresa_id, codigo_sap, codigo_producto, codigo_cliente, descripcion, familia, subfamilia, talla, color, estado_producto",
        "order": "codigo_producto",
        "keywords": ["producto", "productos", "articulo", "sku", "item", "referencia", "codigo_sap", "familia", "subfamilia", "talla", "color"],
    },
    "codigo_barras": {
        "desc": "Códigos de barra (EAN13) asociados a productos.",
        "cols": "id, empresa_id, codigo_barra_ref, codigo_producto, codigo_cliente, cb_ean13, estado",
        "order": "codigo_producto",
        "keywords": ["codigo_barras", "barcode", "ean", "ean13", "codigo de barra"],
    },
    "clientes": {
        "desc": "Clientes registrados en el WMS (personas o empresas compradoras).",
        "cols": "id, id_empresa, rut_cliente, digito_verificador, nombre_cliente, codigo_cliente, email, telefono, fecha_creacion",
        "order": "nombre_cliente",
        "keywords": ["cliente", "clientes", "comprador", "rut", "destinatario"],
    },
    "cliente_direcciones": {
        "desc": "Direcciones de despacho y fiscales de los clientes.",
        "cols": "id, id_cliente, tipo_direccion, direccion, comuna, ciudad, pais, es_activa",
        "order": "id_cliente",
        "keywords": ["direccion", "direcciones", "domicilio", "comuna", "ciudad", "despacho direccion"],
    },
    "proveedores": {
        "desc": "Proveedores de mercancía. Contiene RUT, nombre, dirección y datos de contacto.",
        "cols": "id, empresa_id, prv_id, prv_nombre, prv_correo, prv_telefono1, prv_direccion",
        "order": "prv_nombre",
        "keywords": ["proveedor", "proveedores", "supplier", "fabricante"],
    },
    "maestro_clientes": {
        "desc": "Maestro de clientes retail/tiendas (Falabella, Paris, etc.) con datos EDI.",
        "cols": "id, empresa_id, cli_id, cli_nombre, cli_rut_cliente, cli_tipo_cliente, cli_tienda, cli_comuna, cli_ciudad",
        "order": "cli_nombre",
        "keywords": ["falabella", "paris", "retail", "maestro cliente", "tienda cliente", "cli_nombre"],
    },
    "maestro_tiendas": {
        "desc": "Catálogo de tiendas con su código SAP, bodega WMS y tipo (retail/propia).",
        "cols": "id, empresa_id, id_tienda_sap, tie_codigo, tie_nombre, tie_direccion, tie_comuna, tie_ciudad, tie_estado, tie_es_retail",
        "order": "tie_nombre",
        "keywords": ["tienda", "tiendas", "local", "sucursal", "bodega tienda"],
    },
    "empresas": {
        "desc": "Empresas clientes del WMS (multiempresa). Contiene RUT, nombre, plan activo.",
        "cols": "id, nombre, rut, nombre_legal, nombre_comercial, email, activa, estado",
        "order": "nombre",
        "keywords": ["empresa", "empresas", "compañia", "tenant"],
    },
    "transportistas": {
        "desc": "Transportistas registrados para despacho.",
        "cols": "id, empresa_id, codigo, nombre, rut, telefono, email, estado",
        "order": "nombre",
        "keywords": ["transportista", "transporte", "carrier", "courier empresa"],
    },
    "choferes": {
        "desc": "Choferes asociados a transportistas.",
        "cols": "id, empresa_id, transportista_id, codigo, rut, nombre, apellidos, licencia_conducir, estado",
        "order": "nombre",
        "keywords": ["chofer", "choferes", "conductor", "driver"],
    },
    "catalogos_productos": {
        "desc": "Catálogo de códigos y valores permitidos para campos de productos.",
        "cols": "id, empresa_id, campo_original, campo_interno, codigo, descripcion",
        "order": "campo_interno",
        "keywords": ["catalogo producto", "catalogos productos", "codigos producto"],
    },
    "catalogos_proveedores": {
        "desc": "Catálogo de códigos y valores permitidos para campos de proveedores.",
        "cols": "id, empresa_id, campo_original, campo_interno, codigo, descripcion",
        "order": "campo_interno",
        "keywords": ["catalogo proveedor", "catalogos proveedores"],
    },

    # Stock e inventario
    "stock_wms": {
        "desc": "Stock actual en WMS por producto: cantidad existente, reservada y disponible.",
        "cols": "id, codigo_producto, codigo_cliente, cantidad_existente, cantidad_reservada, cantidad_disponible, fecha_actualizacion, id_empresa",
        "order": "codigo_producto",
        "keywords": ["stock", "inventario", "disponible", "cantidad", "bodega", "existencia", "stock_wms"],
    },
    "stock": {
        "desc": "Stock detallado por producto, ubicación física y lote. Columna 'disponible' calculada.",
        "cols": "id, empresa_id, id_producto, id_ubicacion, lote, fecha_vencimiento, cantidad, reservado, disponible",
        "order": "id_producto",
        "keywords": ["stock ubicacion", "stock lote", "disponible lote", "stock fisico"],
    },
    "stock_reservas": {
        "desc": "Reservas de stock (cantidades comprometidas no despachasdas aún).",
        "cols": "id, codigo_producto, codigo_cliente, cantidad_reservada, fecha_archivo",
        "order": "codigo_producto",
        "keywords": ["reserva", "reservas", "reservado", "stock reservado"],
    },
    "movimientos_stock": {
        "desc": "Historial de movimientos de stock: entradas, salidas, ajustes, traslados y devoluciones.",
        "cols": "id, empresa_id, id_producto, tipo_movimiento, cantidad, stock_anterior, stock_posterior, lote, created_at",
        "order": "created_at DESC",
        "keywords": ["movimiento stock", "entrada stock", "salida stock", "ajuste stock", "traslado stock"],
    },
    "inventario_movimientos": {
        "desc": "Cabecera de movimientos de inventario enviados a SAP (ajustes y traslados internos).",
        "cols": "id, empresa_id, tipo_movimiento, movimiento_tipo, estado, fecha_movimiento, comentarios",
        "order": "fecha_movimiento DESC",
        "keywords": ["movimiento", "movimientos", "traslado", "ajuste inventario"],
    },
    "inventario_movimientos_detalle": {
        "desc": "Detalle (líneas) de movimientos de inventario: producto, cantidad, bodega origen/destino.",
        "cols": "id, movimiento_id, empresa_id, item_code, quantity, warehouse_from, warehouse_to, batch_number",
        "order": "movimiento_id DESC",
        "keywords": ["detalle movimiento", "linea movimiento"],
    },

    # Preparación y despacho
    "preparacion_kp": {
        "desc": "Registros de preparación (picking) de pedidos en el WMS. Incluye número de preparación, pedido, producto, cantidad y preparador.",
        "cols": "id, id_empresa, rut_cliente, numero_preparacion, fecha_preparacion, numero_pedido, codigo_barras, descripcion, codigo_producto, cantidad, cantidad_preparada, cantidad_pendiente, estado_manifiesto, nombre_preparador",
        "order": "fecha_preparacion DESC",
        "keywords": ["preparacion", "picking", "kp", "preparar", "preparado", "cantidad preparada"],
    },
    "preparacion_archivos": {
        "desc": "Archivos de preparación procesados (nombre, fecha, estado, registros).",
        "cols": "id, id_empresa, nombre_archivo, fecha_archivo, fecha_procesado, estado, registros_procesados",
        "order": "fecha_procesado DESC",
        "keywords": ["archivo preparacion", "preparacion archivo"],
    },
    "pedidos_despacho": {
        "desc": "Pedidos de despacho recibidos desde SAP. Estados: 01=Notificado, 02=Inicio picking, 03=Picking parcial, 04=Picking total, 05=Quiebre, 06=Cancelado.",
        "cols": "id, empresa_id, order_number, reference, order_date, customer_code, customer_name, shipping_address, commune, city, sales_channel, courier, status, created_at",
        "order": "order_date DESC",
        "keywords": ["pedido", "pedidos", "despacho", "despachos", "envio", "order", "picking pedido"],
    },
    "detalle_despacho": {
        "desc": "Líneas de detalle de los pedidos de despacho: producto, SKU, cantidad solicitada vs preparada.",
        "cols": "id, empresa_id, pedido_despacho_id, internal_code, sku, total_quantity, picked_quantity, status",
        "order": "pedido_despacho_id DESC",
        "keywords": ["detalle despacho", "items despacho", "linea despacho"],
    },
    "manifiesto": {
        "desc": "Manifiestos de despacho generados. Incluye transportista, chofer, total de bultos y pallets.",
        "cols": "id, numero_manifiesto, fecha_despacho, cantidad_pallet, total_bultos, total_clientes, total_pedidos, courier, transportista_id, chofer_id",
        "order": "fecha_despacho DESC",
        "keywords": ["manifiesto", "manifiestos", "despacho masivo"],
    },
    "manifiesto_detalle": {
        "desc": "Líneas del manifiesto: pedido, código de barras, cantidad preparada.",
        "cols": "id, manifiesto_id, linea, rut_cliente, numero_pedido, codigo_barras, descripcion, cantidad_preparada, lote",
        "order": "manifiesto_id DESC",
        "keywords": ["detalle manifiesto", "linea manifiesto"],
    },
    "kp_despacho_ln": {
        "desc": "Archivo de despacho LN (línea neutra) con folio, número de despacho, SKU y cantidad.",
        "cols": "id, id_empresa, documento, tipo_documento, fecha, numero_despacho, folio, sku, cantidad",
        "order": "fecha DESC",
        "keywords": ["despacho ln", "kp despacho", "folio despacho"],
    },

    # Recepción y OC
    "kp_recepcion": {
        "desc": "Recepciones KP del WMS: número de recepción, SKU, cantidades pedidas/recibidas/aceptadas, operario.",
        "cols": "id, id_empresa, nombre_cliente, numero_recepcion, sku, descripcion, lote, cantidad_pedido, cantidad_recibida, cantidad_aceptada, fecha_operacion",
        "order": "fecha_operacion DESC",
        "keywords": ["recepcion", "recepciones", "recibido", "kp recepcion", "recepcion mercaderia"],
    },
    "kp_recepcion_archivos": {
        "desc": "Archivos de recepción procesados.",
        "cols": "id, id_empresa, nombre_archivo, fecha_procesado, estado, registros_procesados",
        "order": "fecha_procesado DESC",
        "keywords": ["archivo recepcion", "recepcion archivo"],
    },
    "oc": {
        "desc": "Órdenes de compra (OC). Contiene proveedor, cantidades pedidas/recibidas/faltantes, estado de recepción y fechas.",
        "cols": "id, empresa_id, ped_orden_de_compra, ped_rut_proveedor, ped_proveedor, ped_numero_items, ped_cantidad_total_p, ped_cantidad_total_r, ped_estado_recepcion, ped_fecha_creacion, ped_fecha_agendamiento",
        "order": "ped_fecha_creacion DESC",
        "keywords": ["oc", "orden de compra", "ordenes de compra", "compra", "compras"],
    },
    "detalle_oc": {
        "desc": "Líneas de detalle de las órdenes de compra: producto, cantidades pedidas/recibidas/faltantes.",
        "cols": "id, empresa_id, oc_id, ctp_producto, ctp_cantidad_pedida, ctp_cantidad_recibida, ctp_cantidad_faltante",
        "order": "oc_id DESC",
        "keywords": ["detalle oc", "linea oc", "detalle orden compra"],
    },
    "kp_oc": {
        "desc": "Archivo KP de orden de compra con estado, cliente, producto y cantidad.",
        "cols": "id, id_empresa, numero_documento, estado, descripcion_cliente, codigo_producto, sku, descripcion_producto, cantidad, fecha_pedido",
        "order": "fecha_pedido DESC",
        "keywords": ["kp oc", "oc kp", "archivo oc"],
    },
    "recepciones": {
        "desc": "Cabecera de recepciones físicas de OC en el WMS. Estado: PENDIENTE, EN_PROCESO, COMPLETADA, ANULADA.",
        "cols": "id, empresa_id, id_oc, fecha_recepcion, hora_inicio, hora_fin, id_muelle, estado, sap_notification_status",
        "order": "fecha_recepcion DESC",
        "keywords": ["recepcion oc", "recepcion fisica", "recepcion sap"],
    },
    "recepcion_detalle": {
        "desc": "Detalle de recepción física por producto: cantidad recibida, lote, vencimiento y operario.",
        "cols": "id, id_recepcion, id_producto, cantidad_recibida, cantidad_validada, lote, fecha_vencimiento, estado",
        "order": "id_recepcion DESC",
        "keywords": ["detalle recepcion", "recepcion detalle"],
    },

    # APIs e integraciones
    "apis_logs": {
        "desc": "Log de llamadas a APIs externas: endpoint, método, código de respuesta, tiempo de respuesta.",
        "cols": "id, api_id, endpoint, metodo, status_code, response_time, fecha",
        "order": "fecha DESC",
        "keywords": ["api", "apis", "log", "logs", "llamada api", "request", "endpoint", "api log"],
    },
    "apis_config": {
        "desc": "Configuración de APIs externas: endpoint, método, autenticación, estado.",
        "cols": "id, empresa_id, nombre, endpoint, metodo, tipo_autenticacion, estado, descripcion",
        "order": "nombre",
        "keywords": ["config api", "configuracion api", "api config", "apis configuradas"],
    },
    "sap_integration_log": {
        "desc": "Log de integración SAP: operación, dirección (SALIDA/ENTRADA), estado HTTP, éxito/error.",
        "cols": "id, empresa_id, direccion, tipo_operacion, endpoint, http_status, success, error_message, id_referencia, tipo_referencia, created_at",
        "order": "created_at DESC",
        "keywords": ["sap", "integracion sap", "sap log", "log sap"],
    },
    "integraciones_log": {
        "desc": "Log genérico de integraciones del WMS por módulo y acción.",
        "cols": "id, empresa_id, modulo, accion, referencia, referencia_id, usuario_id, fecha",
        "order": "fecha DESC",
        "keywords": ["integracion log", "log integracion", "modulo log"],
    },
    "oms_webhook_logs": {
        "desc": "Logs de webhooks OMS recibidos: order_id, tracking_number, estado procesado.",
        "cols": "id, order_id, remote_order_id, tracking_number, received_at, processed",
        "order": "received_at DESC",
        "keywords": ["webhook", "oms", "oms webhook", "tracking"],
    },
    "exportaciones_historial": {
        "desc": "Historial de exportaciones de archivos: nombre, registros, estado y destino (FTP/descarga).",
        "cols": "id, empresa_id, nombre_archivo, total_registros, procesados, exportados, errores, destino, estado, created_at",
        "order": "created_at DESC",
        "keywords": ["exportacion", "exportaciones", "exportado", "archivo exportado"],
    },
    "importaciones_historial": {
        "desc": "Historial de importaciones de archivos: nombre, registros, estado y errores.",
        "cols": "id, empresa_id, nombre_archivo, total_lineas, exitosos, actualizados, errores, estado, created_at",
        "order": "created_at DESC",
        "keywords": ["importacion", "importaciones", "importado", "archivo importado"],
    },

    # Citas y muelles
    "citas": {
        "desc": "Citas agendadas en muelles de carga/descarga: fecha, hora, vehículo, estado.",
        "cols": "id_cita, id_empresa, id_muelle, fecha_cita, hora_inicio, hora_fin, tipo_servicio, tipo_vehiculo, patente_vehiculo, estado",
        "order": "fecha_cita DESC",
        "keywords": ["cita", "citas", "agendamiento", "muelle cita", "agenda"],
    },
    "bloqueos_horarios": {
        "desc": "Bloqueos de horarios en muelles (feriados, mantenimiento, etc.).",
        "cols": "id_bloqueo, tipo_bloqueo, fecha_inicio, fecha_fin, hora_inicio, hora_fin, id_muelle, motivo, activo",
        "order": "fecha_inicio DESC",
        "keywords": ["bloqueo", "bloqueos", "feriado", "mantenimiento muelle"],
    },
    "historial_estados": {
        "desc": "Auditoría de cambios de estado en cualquier tabla del WMS.",
        "cols": "id, tabla, registro_id, estado_anterior, estado_nuevo, motivo, usuario_id, fecha",
        "order": "fecha DESC",
        "keywords": ["historial estado", "cambio estado", "auditoria"],
    },
    "api_users": {
        "desc": "Usuarios del sistema con sus roles (admin, integracion, consulta, sap).",
        "cols": "id, empresa_id, username, nombre, email, rol, estado, ultimo_login",
        "order": "nombre",
        "keywords": ["usuario", "usuarios", "user", "login usuario", "rol"],
    },
}

ALL_TABLES = list(TABLE_CATALOG.keys())

# ─────────────────────────────────────────────────────────────────────────────
# 2. KEYWORDS GLOBALES PARA DETECTAR PREGUNTAS DE BASE DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

DB_KEYWORDS = [
    # Verbos de consulta
    "cuanto", "cuantos", "cuanta", "cuantas", "cuales", "cual",
    "dame", "muestra", "muestrame", "listar", "lista", "mostrar",
    "trae", "traeme", "busca", "encuentra", "filtra", "consulta",
    "hay", "tiene", "tienen", "existe", "existen", "ver", "veo",
    # Operaciones
    "total", "contar", "suma", "promedio", "maximo", "minimo",
    "mayor", "menor", "mas", "menos", "ultimo", "primero", "reciente",
    "nuevos", "nuevo", "registro", "registros", "cantidad",
    # Entidades del dominio
    "stock", "inventario", "disponible", "reserva", "movimiento",
    "producto", "productos", "sku", "articulo", "codigo",
    "cliente", "clientes", "proveedor", "proveedores",
    "pedido", "pedidos", "despacho", "recepcion", "picking", "preparacion",
    "exportacion", "importacion", "tienda", "tiendas", "transportista",
    "chofer", "manifiesto", "oc", "orden de compra", "cita", "muelle",
    "api", "log", "logs", "sap", "webhook", "empresa",
]

INTERNAL_PROMPTS = [
    "### task:", "suggest 3-5", "generate a concise", "generate 1-3",
    "follow_ups", "json format", "title with an emoji", "chat history",
    "conversation title",
]


def is_db_question(question: str) -> bool:
    q = question.lower().strip()
    if any(p in q for p in INTERNAL_PROMPTS):
        return False
    return any(kw in q for kw in DB_KEYWORDS)


def get_relevant_tables(question: str) -> list[str]:
    q = question.lower().strip()
    relevant = set()
    for table, meta in TABLE_CATALOG.items():
        if any(kw in q for kw in meta["keywords"]):
            relevant.add(table)
    if not relevant:
        # Fallback a tablas más consultadas
        relevant = {"productos", "stock_wms", "clientes", "pedidos_despacho", "preparacion_kp"}
    return list(relevant)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONEXIÓN BD Y LLM
# ─────────────────────────────────────────────────────────────────────────────

def clean_env(value: str, name: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", value).strip()
    if cleaned != value:
        logger.warning(f"Variable {name} fue limpiada de caracteres inválidos")
    return cleaned


db_host     = clean_env(os.getenv("MYSQL_HOST",     "10.110.77.145"), "MYSQL_HOST")
db_port     = clean_env(os.getenv("MYSQL_PORT",     "3306"),          "MYSQL_PORT")
db_user     = clean_env(os.getenv("MYSQL_USER",     "invisia"),       "MYSQL_USER")
db_password = clean_env(os.getenv("MYSQL_PASSWORD", "Vper1821317@"),  "MYSQL_PASSWORD")
db_name     = clean_env(os.getenv("MYSQL_DATABASE", "invisia_db"),    "MYSQL_DATABASE")
ollama_host = clean_env(os.getenv("OLLAMA_HOST", "ollama-phi3.ia.svc.cluster.local"), "OLLAMA_HOST")
ollama_port = clean_env(os.getenv("OLLAMA_PORT", "11434"),            "OLLAMA_PORT")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

missing = [v for v, val in [
    ("MYSQL_HOST", db_host), ("MYSQL_USER", db_user),
    ("MYSQL_PASSWORD", db_password), ("MYSQL_DATABASE", db_name),
] if not val]
if missing:
    raise ValueError(f"Faltan variables de entorno: {', '.join(missing)}")

db_url = f"mysql+pymysql://{db_user}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"
logger.info(f"Conectando a BD: mysql+pymysql://{db_user}:***@{db_host}:{db_port}/{db_name}")

try:
    engine = create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("✅ Conexión a BD exitosa")
except SQLAlchemyError as e:
    logger.error(f"❌ Error conectando a BD: {e}")
    raise

llm = OllamaLLM(
    model=ollama_model,
    base_url=f"http://{ollama_host}:{ollama_port}",
    temperature=0.0,
    num_predict=1500,
    top_k=1,
    stop=None,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERACIÓN DE ESQUEMA ENRIQUECIDO PARA EL PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_context(tables: list[str]) -> str:
    """Construye un contexto de esquema compacto y descriptivo para el LLM."""
    lines = []
    for t in tables:
        meta = TABLE_CATALOG.get(t, {})
        desc = meta.get("desc", "Tabla sin descripción")
        cols = meta.get("cols", "*")
        lines.append(f"-- Tabla: {t}\n-- Descripción: {desc}\n-- Columnas principales: {cols}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERACIÓN Y VALIDACIÓN DE SQL
# ─────────────────────────────────────────────────────────────────────────────

def extract_sql(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw).strip()
    # Buscar el primer SELECT completo hasta ; o fin de texto
    match = re.search(r"(SELECT\b.*?)(?:;|$)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # Evitar doble SELECT
        if re.match(r"(?i)^SELECT\s+SELECT\b", sql):
            sql = sql[7:].strip()
        return sql + ";"
    return ""


def validate_sql(sql: str) -> bool:
    sql = sql.strip().upper()
    if not sql.startswith("SELECT"):
        return False
    # Rechazar operaciones peligrosas
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE"]
    for f in forbidden:
        if re.search(rf"\b{f}\b", sql):
            return False
    return True


def fallback_sql(question: str) -> str:
    q = question.lower()
    for kw, (table, order_col) in [
        ("producto",      ("productos",       "codigo_producto")),
        ("stock",         ("stock_wms",        "codigo_producto")),
        ("cliente",       ("clientes",         "nombre_cliente")),
        ("proveedor",     ("proveedores",      "prv_nombre")),
        ("pedido",        ("pedidos_despacho", "order_date DESC")),
        ("despacho",      ("pedidos_despacho", "order_date DESC")),
        ("preparacion",   ("preparacion_kp",   "fecha_preparacion DESC")),
        ("recepcion",     ("kp_recepcion",      "fecha_operacion DESC")),
        ("oc",            ("oc",               "ped_fecha_creacion DESC")),
        ("manifiesto",    ("manifiesto",       "fecha_despacho DESC")),
        ("movimiento",    ("inventario_movimientos", "fecha_movimiento DESC")),
        ("exportacion",   ("exportaciones_historial", "created_at DESC")),
        ("importacion",   ("importaciones_historial", "created_at DESC")),
        ("transportista", ("transportistas",   "nombre")),
        ("chofer",        ("choferes",         "nombre")),
        ("tienda",        ("maestro_tiendas",  "tie_nombre")),
        ("log",           ("apis_logs",        "fecha DESC")),
        ("api",           ("apis_logs",        "fecha DESC")),
        ("sap",           ("sap_integration_log", "created_at DESC")),
        ("empresa",       ("empresas",         "nombre")),
        ("cita",          ("citas",            "fecha_cita DESC")),
    ]:
        if kw in q:
            if any(x in q for x in ["cuanto", "contar", "total", "cantidad"]):
                return f"SELECT COUNT(*) AS total FROM {table};"
            return f"SELECT * FROM {table} ORDER BY {order_col} LIMIT 20;"
    return "SELECT 'No se pudo generar la consulta. Por favor reformula la pregunta.' AS mensaje;"


SQL_SYSTEM_PROMPT = """Eres un generador de consultas SQL para MySQL 8 en un sistema WMS (Warehouse Management System).

REGLAS ABSOLUTAS:
1. Responde ÚNICAMENTE con la sentencia SQL SELECT. Sin explicaciones, sin markdown, sin comillas extras.
2. NUNCA uses INSERT, UPDATE, DELETE, DROP, TRUNCATE o ALTER.
3. Usa siempre LIMIT (máximo 50 si el usuario no especifica cantidad).
4. Para preguntas de conteo usa COUNT(*) o COUNT(columna) con alias descriptivo.
5. Usa ORDER BY lógico: fechas más recientes primero (DESC), nombres en orden alfabético.
6. Para JOINs usa alias cortos (p=productos, s=stock_wms, c=clientes, etc.).
7. Si la pregunta pide "todos" o no especifica límite, usa LIMIT 50.
8. Para búsquedas por nombre usa LIKE '%texto%' (case insensitive por defecto en MySQL).
9. Selecciona columnas específicas relevantes, NO uses SELECT * excepto cuando el usuario pida "todo".
10. Si necesitas unir tablas, hazlo correctamente con JOIN.

CONTEXTO DEL NEGOCIO:
- stock_wms.cantidad_disponible = stock disponible para venta
- pedidos_despacho.status: 01=Notificado, 02=Inicio picking, 03=Picking parcial, 04=Picking total, 05=Quiebre, 06=Cancelado
- inventario_movimientos.tipo_movimiento: 01=Ajuste, 02=Traslado interno
- maestro_tiendas.tie_es_retail: Y=Falabella/retail, N=tienda propia
- productos.procedencia: 0=Nacional, 1=Importado

ESQUEMA DE TABLAS RELEVANTES:
{schema}

PREGUNTA: {question}
SQL:"""


def get_sql_with_retry(question: str, schema: str, max_retries: int = 3) -> str:
    prompt = SQL_SYSTEM_PROMPT.format(schema=schema, question=question)

    for attempt in range(max_retries):
        raw = llm.invoke(prompt)
        logger.info(f"[SQL intento {attempt+1}] raw: {raw[:200]!r}")
        sql = extract_sql(raw)
        if sql and validate_sql(sql):
            logger.info(f"[SQL válido] {sql}")
            return sql
        logger.warning(f"[SQL intento {attempt+1}] inválido, reintentando...")
        prompt = (
            f"IMPORTANTE: Solo escribe la sentencia SELECT para MySQL, sin texto adicional.\n"
            f"Pregunta: {question}\n"
            f"Tablas disponibles: {', '.join(get_relevant_tables(question))}\n"
            f"SQL:"
        )

    logger.warning("Usando fallback SQL")
    return fallback_sql(question)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FORMATEO INTELIGENTE DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def serialize_row(row: dict) -> dict:
    """Convierte valores no serializables (datetime, Decimal) a string."""
    result = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        elif hasattr(v, "__float__"):
            result[k] = float(v)
        else:
            result[k] = v
    return result


def format_row_for_display(row: dict) -> str:
    """Extrae la representación más legible de una fila."""
    priority_keys = [
        "nombre", "nombre_cliente", "cli_nombre", "tie_nombre", "prv_nombre",
        "descripcion", "descripcion_producto", "nombre_producto",
        "codigo_producto", "sku", "codigo_sap",
        "order_number", "numero_preparacion", "numero_recepcion",
        "numero_manifiesto", "ped_orden_de_compra",
    ]
    parts = []
    # Primero agregar el campo de nombre/descripción principal
    for key in priority_keys:
        if key in row and row[key]:
            parts.append(f"{row[key]}")
            break
    # Agregar campos secundarios relevantes (max 3)
    secondary = ["cantidad", "cantidad_disponible", "status", "estado", "fecha",
                 "order_date", "fecha_preparacion", "fecha_operacion", "total_registros"]
    added = 0
    for key in secondary:
        if key in row and row[key] is not None and added < 3:
            parts.append(f"{key}={row[key]}")
            added += 1
    # Si no encontramos nada, tomar los primeros 2 valores
    if not parts:
        vals = [str(v) for v in list(row.values())[:2] if v is not None]
        parts = vals
    return " | ".join(parts)


ANALYSIS_PROMPT = """Eres el asistente del sistema WMS de la empresa. Analiza los resultados de una consulta y responde en español de forma clara, ordenada y útil para un operador de bodega o gerente.

Pregunta original: {question}
SQL ejecutado: {sql}
Total de registros encontrados: {total}
Muestra de datos ({sample_size} registros):
{sample_text}

INSTRUCCIONES DE RESPUESTA:
1. Responde directamente a la pregunta, en español.
2. Muestra el total de registros encontrados.
3. Si son listados: usa numeración y muestra los datos más relevantes (nombre, código, cantidad, estado).
4. Si es un conteo: muestra el número de forma destacada.
5. Si hay fechas, ordénalas cronológicamente en el listado.
6. Agrega un breve análisis: tendencia, alerta o conclusión útil.
7. Si hay campos de estado, explica qué significa cada estado que aparezca.
8. Sé conciso pero completo. No uses lenguaje técnico innecesario.

Respuesta:"""


def format_and_analyze(question: str, sql: str, rows: list[dict]) -> str:
    total = len(rows)
    if total == 0:
        return "✅ La consulta se ejecutó correctamente pero **no se encontraron registros** que coincidan con los criterios."

    # Si es un resultado de conteo simple
    if total == 1 and len(rows[0]) == 1:
        val = list(rows[0].values())[0]
        col = list(rows[0].keys())[0]
        if col.lower() in ("total", "count(*)", "count", "cantidad"):
            return f"📊 **Total encontrado:** {val:,} registros."

    sample = [serialize_row(r) for r in rows[:25]]
    sample_text = "\n".join(
        f"{i+1}. {format_row_for_display(r)}" for i, r in enumerate(sample)
    )

    prompt = ANALYSIS_PROMPT.format(
        question=question,
        sql=sql,
        total=total,
        sample_size=len(sample),
        sample_text=sample_text,
    )

    try:
        analysis = llm.invoke(prompt)
        # Limpiar posibles artefactos
        analysis = re.sub(r"```.*?```", "", analysis, flags=re.DOTALL).strip()
        return analysis
    except Exception as e:
        logger.error(f"Error en análisis LLM: {e}")
        # Respuesta de fallback sin LLM
        lines = [f"Se encontraron **{total}** registros.\n"]
        if total > 0:
            lines.append("Primeros resultados:")
            for i, row in enumerate(sample[:10], 1):
                lines.append(f"  {i}. {format_row_for_display(row)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class Query(BaseModel):
    query: str
    empresa_id: int | None = None   # Opcional: filtrar por empresa
    limit: int | None = None        # Opcional: forzar límite de filas


@app.post("/generate")
async def generate(query: Query):
    q = query.query.strip()
    logger.info(f"{'='*60}\nPregunta: {q}")

    if not q:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    # ── Modo consulta BD ──────────────────────────────────────────
    if is_db_question(q):
        logger.info("Modo: consulta BD")
        try:
            relevant = get_relevant_tables(q)
            logger.info(f"Tablas relevantes: {relevant}")

            schema = build_schema_context(relevant)

            # Enriquecer pregunta con contexto de empresa si se provee
            enriched_q = q
            if query.empresa_id:
                enriched_q = f"{q} [filtrar solo empresa_id={query.empresa_id}]"
            if query.limit:
                enriched_q = f"{enriched_q} [mostrar máximo {query.limit} resultados]"

            sql = get_sql_with_retry(enriched_q, schema)

            # Ejecutar
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]

            logger.info(f"Filas obtenidas: {len(rows)}")
            answer = format_and_analyze(q, sql, rows)
            logger.info(f"Respuesta generada OK ({len(answer)} chars)")

            return {
                "response": answer,
                "meta": {
                    "sql": sql,
                    "rows_returned": len(rows),
                    "tables_used": relevant,
                },
            }

        except SQLAlchemyError as e:
            logger.error(f"Error SQL: {e}")
            raise HTTPException(status_code=500, detail=f"Error al ejecutar la consulta SQL: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # ── Modo conversación general ──────────────────────────────────
    else:
        logger.info("Modo: conversación general")
        try:
            chat_prompt = (
                "Eres el Asistente WMS de la empresa, un experto en gestión de bodegas, "
                "logística y operaciones de warehouse. Respondes en español de forma concisa "
                "y profesional.\n\n"
                f"Usuario: {q}\n\n"
                "Asistente:"
            )
            raw = llm.invoke(chat_prompt)
            # Limpiar markdown y respuestas excesivamente largas
            answer = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
            # Cortar si hay doble salto de línea innecesario al inicio
            answer = answer.lstrip("\n")
            logger.info(f"Respuesta chat ({len(answer)} chars)")
            return {"response": answer, "meta": {"mode": "chat"}}

        except Exception as e:
            logger.error(f"Error chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Verifica conexión a BD y LLM."""
    bd_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        bd_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if bd_ok else "degraded",
        "db": "connected" if bd_ok else "error",
        "model": ollama_model,
        "total_tables": len(ALL_TABLES),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/tables")
async def list_tables():
    """Devuelve el catálogo completo de tablas con sus descripciones."""
    return {
        t: {"description": m["desc"], "key_columns": m["cols"]}
        for t, m in TABLE_CATALOG.items()
    }


@app.get("/tables/{table_name}")
async def table_info(table_name: str):
    """Información de una tabla específica."""
    if table_name not in TABLE_CATALOG:
        raise HTTPException(status_code=404, detail=f"Tabla '{table_name}' no encontrada en el catálogo.")
    meta = TABLE_CATALOG[table_name]
    try:
        with engine.connect() as conn:
            res = conn.execute(text(f"SELECT COUNT(*) AS total FROM `{table_name}`"))
            count = res.fetchone()[0]
    except Exception:
        count = None
    return {
        "table": table_name,
        "description": meta["desc"],
        "key_columns": meta["cols"],
        "keywords": meta["keywords"],
        "total_rows": count,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
