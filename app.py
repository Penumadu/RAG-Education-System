"""
Streamlit Web UI for Strictly-Grounded PDF RAG Application.
Google Gemini-inspired interface:
- Ultra-clean centered reading canvas (max-width 860px)
- Zero top clutter: sleek Gemini top bar & model pill
- Floating capsule input bar with generous multiline typing area
- Unobstructed, high-legibility answer presentation with visual Mermaid diagrams
- Interactive 3D Vector mesh neatly housed in the sidebar expander
"""

from __future__ import annotations

import datetime
import importlib
import json
import os
import re
import uuid
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import agent as _agent_mod
import rag_engine as _rag_engine_mod

# Reload submodules so live edits to rag_engine or agent are picked up immediately
importlib.reload(_rag_engine_mod)
importlib.reload(_agent_mod)

from agent import RAGAgentService
from rag_engine import DocumentProcessingError, RAGEngine, check_ollama_status

load_dotenv()

# --------------- Page configuration ---------------
st.set_page_config(
    page_title="Professor AI • Gemini Classroom",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Authentic Google Gemini Chat UI
st.markdown(
    """
    <style>
    /* Global Typography & Smoothing */
    html, body {
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
        background-color: #131314 !important;
        color: #e3e3e3 !important;
    }

    /* Minimize Streamlit Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.2rem !important;
        z-index: 10 !important;
    }

    /* Google Gemini Full-Height Centered Reading Canvas */
    .block-container {
        max-width: 880px !important;
        padding-top: 0.25rem !important;
        padding-bottom: 6.5rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Gemini Ultra-Sleek Top Bar */
    .gemini-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 0 10px 0;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .gemini-logo {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.25rem;
        font-weight: 600;
        color: #e3e3e3;
        letter-spacing: -0.01em;
    }
    .gemini-sparkle {
        font-size: 1.4rem;
        background: linear-gradient(135deg, #8ab4f8 0%, #c58af9 50%, #f28b82 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .gemini-badge {
        font-size: 0.74rem;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .gemini-badge-model {
        background: rgba(138, 180, 248, 0.12);
        color: #8ab4f8;
        border: 1px solid rgba(138, 180, 248, 0.28);
    }
    .gemini-badge-docs {
        background: rgba(52, 168, 83, 0.12);
        color: #81c995;
        border: 1px solid rgba(52, 168, 83, 0.28);
    }

    /* Authentic Google Gemini Welcome Greeting */
    .gemini-hero {
        margin-top: 48px;
        margin-bottom: 32px;
    }
    .gemini-hero-title {
        font-size: 3.0rem;
        font-weight: 500;
        background: linear-gradient(74deg, #4285f4 0%, #9b72cb 18%, #d96570 38%, #8ab4f8 85%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1.18;
        letter-spacing: -0.02em;
    }
    .gemini-hero-sub {
        font-size: 1.65rem;
        font-weight: 400;
        color: #8e918f;
        margin-top: 0;
        line-height: 1.35;
    }

    /* Gemini Suggestion Prompt Cards */
    div[data-testid="stButton"] > button {
        border-radius: 16px !important;
        background: #1e1f20 !important;
        border: 1px solid #333537 !important;
        color: #e3e3e3 !important;
        padding: 16px 18px !important;
        font-size: 0.95rem !important;
        line-height: 1.45 !important;
        text-align: left !important;
        height: auto !important;
        min-height: 78px !important;
        transition: all 0.2s cubic-bezier(0.2, 0, 0, 1) !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #282a2c !important;
        border-color: #8ab4f8 !important;
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
    }

    /* Sidebar Search History Buttons & Actions */
    div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        min-height: 38px !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        font-size: 0.88rem !important;
        line-height: 1.35 !important;
        background: #1e1f20 !important;
        border: 1px solid #333537 !important;
        justify-content: flex-start !important;
        text-align: left !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: none !important;
        transform: none !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: #282a2c !important;
        border-color: #8ab4f8 !important;
        color: #8ab4f8 !important;
        transform: none !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
        background: #8ab4f8 !important;
        color: #131314 !important;
        font-weight: 600 !important;
        border: none !important;
        justify-content: center !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #a8c7fa !important;
        color: #131314 !important;
    }

    /* Gemini Chat Messages: User Message Bubble (Right-Aligned Pill) */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: transparent !important;
        flex-direction: row-reverse !important;
        gap: 12px !important;
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:nth-child(2) {
        background: #282a2c !important;
        border: 1px solid #3c4043 !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 12px 20px !important;
        max-width: 82% !important;
        color: #f1f5f9 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stChatMessageAvatarUser"] {
        background: #3c4043 !important;
        color: #8ab4f8 !important;
        border-radius: 50% !important;
    }

    /* Gemini Chat Messages: Assistant Message (Expansive Full-Width Stream) */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-top: 0.6rem !important;
        margin-bottom: 2.2rem !important;
        gap: 16px !important;
    }
    div[data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #4285f4 0%, #9b72cb 50%, #d96570 100%) !important;
        border-radius: 50% !important;
        width: 34px !important;
        height: 34px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 10px rgba(66, 133, 244, 0.35) !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:nth-child(2) {
        background: transparent !important;
        padding: 0 !important;
        width: 100% !important;
    }

    /* Gemini Answer Typography & Structure */
    .gemini-answer-text,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] {
        font-size: 1.08rem !important;
        line-height: 1.82 !important;
        color: #e3e3e3 !important;
        letter-spacing: -0.005em;
    }
    .gemini-answer-text h1, .gemini-answer-text h2, .gemini-answer-text h3,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] h3 {
        color: #8ab4f8 !important;
        font-weight: 600 !important;
        margin-top: 22px;
        margin-bottom: 10px;
        letter-spacing: -0.015em;
    }
    .gemini-answer-text h4, .gemini-answer-text h5,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] h5 {
        color: #c58af9 !important;
        font-weight: 600 !important;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    .gemini-answer-text table,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        background: #1e1f20;
        border-radius: 8px;
        overflow: hidden;
    }
    .gemini-answer-text th, .gemini-answer-text td,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] th,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] td {
        border: 1px solid #333537;
        padding: 10px 14px;
        text-align: left;
    }
    .gemini-answer-text th,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] th {
        background: rgba(138, 180, 248, 0.12);
        color: #8ab4f8;
        font-weight: 600;
    }

    /* ================= GEMINI FLOATING INPUT CAPSULE ================= */
    div[data-testid="stBottom"] {
        background: linear-gradient(180deg, rgba(19, 19, 20, 0) 0%, rgba(19, 19, 20, 0.88) 25%, #131314 100%) !important;
        backdrop-filter: blur(12px) !important;
        padding-top: 14px !important;
        padding-bottom: 12px !important;
    }

    div[data-testid="stBottom"]::after {
        content: "Professor AI is strictly grounded in uploaded PDF documents • Always verify important formulas";
        display: block;
        text-align: center;
        color: #8e918f;
        font-size: 0.74rem;
        margin-top: 8px;
        letter-spacing: 0.01em;
        pointer-events: none;
    }

    div[data-testid="stChatInput"] {
        max-width: 860px !important;
        margin: 0 auto !important;
        padding: 0 !important;
    }

    div[data-testid="stChatInput"] > div {
        border-radius: 28px !important;
        border: 1px solid #3c4043 !important;
        background: #1e1f20 !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45) !important;
        transition: all 0.22s ease-in-out !important;
        min-height: 64px !important;
        padding-left: 14px !important;
        padding-right: 8px !important;
    }

    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #8ab4f8 !important;
        background: #282a2c !important;
        box-shadow: 0 0 0 1px #8ab4f8, 0 8px 30px rgba(0, 0, 0, 0.6) !important;
    }

    div[data-testid="stChatInput"] textarea {
        font-size: 1.08rem !important;
        line-height: 1.55 !important;
        min-height: 58px !important;
        max-height: 240px !important;
        padding: 16px 14px !important;
        color: #e3e3e3 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        font-size: 1.04rem !important;
        color: #8e918f !important;
        opacity: 0.95 !important;
    }

    /* Gemini Circular Send Button */
    div[data-testid="stChatInput"] button {
        height: 42px !important;
        width: 42px !important;
        border-radius: 50% !important;
        background: #8ab4f8 !important;
        border: none !important;
        margin-right: 6px !important;
        margin-bottom: 8px !important;
        color: #131314 !important;
        transition: all 0.18s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stChatInput"] button:hover {
        background: #a8c7fa !important;
        transform: scale(1.06) !important;
        box-shadow: 0 0 14px rgba(138, 180, 248, 0.6) !important;
    }

    div[data-testid="stChatInput"] button svg {
        height: 22px !important;
        width: 22px !important;
        fill: #131314 !important;
    }

    /* Gemini Citation Expandable Cards */
    .citation-card {
        background: #1e1f20;
        border: 1px solid #333537;
        border-left: 3px solid #8ab4f8;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .citation-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.8rem;
    }
    .citation-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.74rem;
    }
    .badge-source {
        background: rgba(138, 180, 248, 0.14);
        color: #8ab4f8;
        border: 1px solid rgba(138, 180, 248, 0.3);
    }
    .badge-page {
        background: rgba(255, 255, 255, 0.08);
        color: #c4c7c5;
    }
    .badge-confidence-high {
        background: rgba(52, 168, 83, 0.14);
        color: #81c995;
    }
    .badge-confidence-mid {
        background: rgba(138, 180, 248, 0.14);
        color: #8ab4f8;
    }
    .citation-text {
        color: #c4c7c5;
        font-size: 0.88rem;
        line-height: 1.6;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache version — bump to force singleton re-creation when needed
_CACHE_VERSION = "v19"


@st.cache_resource(show_spinner="Initializing vector store and RAG engine...")
def get_rag_engine(_v: str = _CACHE_VERSION) -> RAGEngine:
    """Initialize singleton RAG engine connected to persistent ChromaDB storage."""
    return RAGEngine(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./.chroma_data"),
        embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


@st.cache_resource(show_spinner="Initializing LangChain agent harness...")
def get_agent_service(_rag_engine: RAGEngine, _v: str = _CACHE_VERSION) -> RAGAgentService:
    """Initialize singleton Agent service with memory checkpointer."""
    return RAGAgentService(
        rag_engine=_rag_engine,
        model_name=os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:9b"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


rag_engine = get_rag_engine(_v=_CACHE_VERSION)

# Guard against stale cached instances: if an old cached RAGEngine lacks new methods, clear cache and rerun
if not hasattr(rag_engine, "get_indexed_documents"):
    st.cache_resource.clear()
    st.rerun()

agent_service = get_agent_service(rag_engine, _v=_CACHE_VERSION)

is_railway = bool(
    os.getenv("RAILWAY_ENVIRONMENT")
    or os.getenv("USE_GEMINI_ONLY", "").lower() in ("true", "1", "yes")
    or os.getenv("PORT")
)
has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

if is_railway:
    # On Railway: strictly Gemini only. Never probe or attempt local Ollama.
    model_service_available = has_gemini
    ollama_online = False
else:
    @st.cache_data(ttl=15)
    def is_ollama_reachable() -> bool:
        return check_ollama_status(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    ollama_online = is_ollama_reachable()
    model_service_available = has_gemini or ollama_online

# --------------- State management ---------------
if "indexed_files" not in st.session_state or not isinstance(st.session_state.get("indexed_files"), dict):
    st.session_state["indexed_files"] = rag_engine.get_indexed_documents()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

if "search_history" not in st.session_state:
    st.session_state["search_history"] = []
    # Seed search history from any existing user messages
    for m in st.session_state.get("messages", []):
        if m.get("role") == "user" and m.get("content"):
            st.session_state["search_history"].append({
                "id": uuid.uuid4().hex[:6],
                "query": m["content"],
                "time": "Recent",
            })


# --------------- Three.js 3D Vector Visualizer (in sidebar) ---------------
def render_threejs_vector_mesh(height: int = 150) -> None:
    """Render an interactive 3D vector space constellation using Three.js in an isolated iframe."""
    three_html = """<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<style>
  body { margin: 0; padding: 0; overflow: hidden; background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  #container { width: 100%; height: 146px; position: relative; border-radius: 10px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.1); background: radial-gradient(circle at 50% 50%, #0e172a 0%, #060a12 100%); }
  .tag { position: absolute; top: 8px; left: 10px; color: #818cf8; font-size: 10px; font-weight: 600; background: rgba(129, 140, 248, 0.15); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(129, 140, 248, 0.3); z-index: 10; pointer-events: none; }
  .hint { position: absolute; bottom: 6px; right: 8px; color: #64748b; font-size: 9px; z-index: 10; pointer-events: none; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
</style>
<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>
</head>
<body>
<div id='container'>
  <div class='tag'>⚡ 3D VECTOR SPACE</div>
  <div class='hint'>Drag to rotate</div>
  <canvas id='c'></canvas>
</div>
<script>
window.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('container');
  const canvas = document.getElementById('c');
  if (!container || !canvas || !window.THREE) return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.z = 28;

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const count = 80;
  const geom = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);

  const cCyan = new THREE.Color(0x38bdf8);
  const cPurple = new THREE.Color(0x818cf8);
  const cPink = new THREE.Color(0xc084fc);

  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 36;
    pos[i * 3 + 1] = (Math.random() - 0.5) * 16;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 18;

    const r = Math.random();
    const c = r > 0.6 ? cCyan : (r > 0.3 ? cPurple : cPink);
    col[i * 3] = c.r;
    col[i * 3 + 1] = c.g;
    col[i * 3 + 2] = c.b;
  }

  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));

  const pMat = new THREE.PointsMaterial({ size: 1.8, vertexColors: true, transparent: true, opacity: 0.85 });
  const points = new THREE.Points(geom, pMat);
  scene.add(points);

  const lineMat = new THREE.LineBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.18 });
  const linePos = [];
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      const dx = pos[i * 3] - pos[j * 3];
      const dy = pos[i * 3 + 1] - pos[j * 3 + 1];
      const dz = pos[i * 3 + 2] - pos[j * 3 + 2];
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (d < 8) {
        linePos.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        linePos.push(pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]);
      }
    }
  }
  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
  const lines = new THREE.LineSegments(lineGeom, lineMat);
  scene.add(lines);

  let mouseX = 0, mouseY = 0, targetX = 0, targetY = 0;
  container.addEventListener('mousemove', (e) => {
    const rect = container.getBoundingClientRect();
    mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
  });

  window.addEventListener('resize', () => {
    if (!container || !renderer || !camera) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  function animate() {
    requestAnimationFrame(animate);
    targetX += (mouseX * 0.45 - targetX) * 0.05;
    targetY += (mouseY * 0.45 - targetY) * 0.05;

    points.rotation.y += 0.002;
    points.rotation.x = targetY * 0.3;
    lines.rotation.y += 0.002;
    lines.rotation.x = targetY * 0.3;

    renderer.render(scene, camera);
  }
  animate();
});
</script>
</body>
</html>"""
    components.html(three_html, height=height)


# --------------- Helper: render citations ---------------
def _render_citations(citations: list, max_chars: int = 400) -> None:
    """Render Gemini-style expandable source citation chips."""
    if not citations:
        return
    with st.expander(f":material/description: Verified Source Citations ({len(citations)})", expanded=False):
        for idx, cit in enumerate(citations, start=1):
            preview = cit.content[:max_chars] + "…" if len(cit.content) > max_chars else cit.content
            relevance_pct = int(cit.score * 100) if cit.score <= 1.0 else int(cit.score)
            conf_class = "badge-confidence-high" if relevance_pct >= 75 else "badge-confidence-mid"

            st.markdown(
                f"""<div class="citation-card">
                    <div class="citation-header">
                        <span class="citation-badge badge-source">📄 #{idx} {cit.source}</span>
                        <span class="citation-badge badge-page">🔖 Page {cit.page}</span>
                        <span class="citation-badge {conf_class}">🎯 {relevance_pct}% match</span>
                    </div>
                    <div class="citation-text">"{preview}"</div>
                </div>""",
                unsafe_allow_html=True,
            )


def _sanitize_mermaid_code(code: str) -> str:
    """Pre-sanitize LLM Mermaid code to eliminate common syntax crashes."""
    clean = code.strip()
    clean = re.sub(r"^```mermaid\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"```$", "", clean).strip()

    lines = [ln.rstrip() for ln in clean.splitlines() if ln.strip()]
    if not lines:
        return "graph TD\n  Start[\"Start\"] --> End[\"End\"]"

    valid_headers = ("graph ", "flowchart ", "sequencediagram", "classdiagram", "statediagram", "erdiagram", "gantt", "pie", "gitgraph")
    first_line_lower = lines[0].strip().lower()
    if not any(first_line_lower.startswith(h) for h in valid_headers):
        lines.insert(0, "graph TD")

    fixed_lines = [lines[0]]
    for line in lines[1:]:
        # Quote node labels if not already quoted
        def _quote_bracket(m):
            node_id = m.group(1)
            b_open = m.group(2)
            content = m.group(3).strip()
            b_close = m.group(4)
            if content.startswith('"') and content.endswith('"'):
                inner = content[1:-1].replace('"', "'")
                return f'{node_id}{b_open}"{inner}"{b_close}'
            safe = content.replace('"', "'").replace("&", "and").replace("<", "lt").replace(">", "gt")
            return f'{node_id}{b_open}"{safe}"{b_close}'

        l_fixed = re.sub(r'([A-Za-z0-9_\-]+)\s*(\[)([^\]\n]+)(\])', _quote_bracket, line)
        l_fixed = re.sub(r'([A-Za-z0-9_\-]+)\s*(\()([^\)\n]+)(\))', _quote_bracket, l_fixed)
        l_fixed = re.sub(r'([A-Za-z0-9_\-]+)\s*(\{)([^\}\n]+)(\})', _quote_bracket, l_fixed)
        fixed_lines.append(l_fixed)

    return "\n".join(fixed_lines)


MERMAID_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      background-color: #1e1f20;
      color: #e3e3e3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 16px 12px;
      border-radius: 12px;
      border: 1px solid #3c4043;
      overflow: auto;
    }
    .diagram-header {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid #333537;
      font-size: 0.82rem;
      color: #9aa0a6;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .diagram-header .title {
      display: flex;
      align-items: center;
      gap: 6px;
      color: #8ab4f8;
      font-weight: 600;
    }
    #mermaid-target {
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 180px;
    }
    svg {
      max-width: 100% !important;
      height: auto !important;
    }
  </style>
</head>
<body>
  <div class="diagram-header">
    <span class="title">📐 Visual Problem Analysis & Relationships</span>
    <span>Visual Diagram</span>
  </div>
  <div id="mermaid-target">
    <div style="color: #9aa0a6; font-size: 13px;">Rendering diagram...</div>
  </div>

  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

    mermaid.initialize({
      startOnLoad: false,
      suppressErrorRendering: true,
      theme: 'dark',
      securityLevel: 'loose',
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      themeVariables: {
        darkMode: true,
        background: '#1e1f20',
        primaryColor: '#282a2c',
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#8ab4f8',
        lineColor: '#c58af9',
        secondaryColor: '#303134',
        tertiaryColor: '#131314',
        mainBkg: '#282a2c',
        nodeBorder: '#8ab4f8',
        textColor: '#e3e3e3'
      }
    });

    const rawCode = __JSON_CODE__;
    const target = document.getElementById('mermaid-target');

    async function renderMermaid() {
      try {
        const renderId = 'm_' + Math.random().toString(36).substring(2, 9);
        const { svg } = await mermaid.render(renderId, rawCode);
        target.innerHTML = svg;
      } catch (err) {
        console.warn("Primary Mermaid render failed:", err);
        try {
          const repaired = rawCode
            .split('\\n')
            .map(l => {
              if (l.trim().match(/^(graph|flowchart|sequenceDiagram|classDiagram)/i)) return l;
              return l.replace(/([\\[\\(\\{])([^"'\\n\\]\\)\\}]+)([\\]\\)\\}])/g, (m, o, t, c) => `${o}"${t.replace(/"/g, "'").replace(/[<>&]/g, ' ')}"${c}`);
            })
            .join('\\n');
          const repairId = 'm_rep_' + Math.random().toString(36).substring(2, 9);
          const { svg } = await mermaid.render(repairId, repaired);
          target.innerHTML = svg;
        } catch (err2) {
          console.warn("Mermaid auto-repair failed, showing clean fallback:", err2);
          const lines = rawCode.split('\\n').filter(l => l.trim() && !l.trim().match(/^(graph|flowchart)/i));
          const items = lines.map(l => {
            const cleanLine = l.replace(/[A-Za-z0-9_\\-]+\\s*[\\[\\(\\{]/g, '').replace(/[\\]\\)\\}]/g, '').replace(/-->/g, ' ➔ ').replace(/["']/g, '').trim();
            return `<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:#282a2c;border:1px solid #3c4043;border-radius:8px;margin-bottom:6px;font-size:13px;color:#e3e3e3;"><span style="color:#8ab4f8;">•</span><span>${cleanLine}</span></div>`;
          }).join('');
          target.innerHTML = `
            <div style="width:100%;text-align:left;padding:8px 0;">
              <div style="color:#8ab4f8;font-weight:600;font-size:13px;margin-bottom:8px;">✦ Problem Logic Flow</div>
              ${items || '<pre style="color:#9aa0a6;">' + rawCode + '</pre>'}
            </div>
          `;
        }
      }
    }

    renderMermaid();
  </script>
</body>
</html>"""


def _render_mermaid_diagram(code: str) -> None:
    """Render a visual Mermaid diagram with Google Gemini dark theme, auto-repair, and error suppression."""
    clean_code = _sanitize_mermaid_code(code)
    json_code = json.dumps(clean_code)

    lines = [line for line in clean_code.splitlines() if line.strip()]
    estimated_height = max(260, min(650, len(lines) * 42 + 130))

    html_code = MERMAID_HTML_TEMPLATE.replace("__JSON_CODE__", json_code)

    st.components.v1.html(html_code, height=estimated_height, scrolling=True)

    with st.expander("🔍 View Diagram Mermaid Source", expanded=False):
        st.code(clean_code, language="mermaid")


def _render_gemini_response(content: str, citations: list) -> None:
    """Render assistant answer, splitting markdown text and visual Mermaid diagrams."""
    pattern = r'(```mermaid\s*[\r\n]+.*?```)'
    parts = re.split(pattern, content, flags=re.DOTALL | re.IGNORECASE)

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if re.match(r"^```mermaid", stripped, flags=re.IGNORECASE):
            _render_mermaid_diagram(stripped)
        else:
            st.markdown(part)

    _render_citations(citations)


active_docs = len(st.session_state["indexed_files"])
total_chunks = rag_engine.document_count

# Capture chat input or pending prompt before rendering sidebar so search history is immediately updated
if not model_service_available:
    placeholder_text = "Local Ollama is offline. Connect your Google Gemini API key above to chat..."
    chat_disabled = True
elif active_docs == 0:
    placeholder_text = "Upload a PDF first via the sidebar to start asking questions..."
    chat_disabled = True
else:
    placeholder_text = "Ask Professor AI anything from your course materials..."
    chat_disabled = False

chat_input_val = st.chat_input(
    placeholder=placeholder_text,
    disabled=chat_disabled,
)

user_query = chat_input_val or (st.session_state.pop("pending_prompt", None))
if user_query:
    now_str = datetime.datetime.now().strftime("%I:%M %p")
    existing_hist = [h for h in st.session_state.get("search_history", []) if h.get("query") != user_query]
    st.session_state["search_history"] = [
        {"id": uuid.uuid4().hex[:6], "query": user_query, "time": now_str}
    ] + existing_hist[:25]

has_conversation = len(st.session_state["messages"]) > 0 or user_query is not None


# ================= SIDEBAR: SEARCH HISTORY & REPOSITORY =================
with st.sidebar:
    # 1. Action Row: New Search & Clear History
    col_new, col_clear = st.columns([3, 1])
    with col_new:
        if st.button(":material/add: New Search", type="primary", width="stretch", help="Start a new search / question"):
            st.session_state["messages"] = []
            st.session_state["thread_id"] = f"session_{uuid.uuid4().hex[:8]}"
            st.rerun()
    with col_clear:
        if st.button(":material/delete_sweep:", help="Clear search history", width="stretch"):
            st.session_state["search_history"] = []
            st.toast("Search history cleared.", icon=":material/delete:")
            st.rerun()

    # 2. User Search History Section
    st.markdown("##### :material/history: Search History")
    search_history = st.session_state.get("search_history", [])

    if search_history:
        if len(search_history) > 4:
            h_filter = st.text_input(
                "Filter history...",
                placeholder="Search history...",
                label_visibility="collapsed",
                key="sidebar_hist_filter",
            ).strip().lower()
            filtered_history = [
                h for h in search_history if h_filter in h["query"].lower()
            ] if h_filter else search_history
        else:
            filtered_history = search_history

        for item in filtered_history[:12]:
            q_text = item["query"]
            clean_display = q_text.replace("\n", " ").strip()
            label = clean_display if len(clean_display) <= 26 else clean_display[:24] + "…"
            t_str = item.get("time", "")

            if st.button(
                f":material/search: {label}",
                key=f"hist_item_{item['id']}",
                help=f"{q_text}\n(Searched at {t_str})",
                width="stretch",
            ):
                st.session_state["pending_prompt"] = q_text
                st.rerun()
    else:
        st.caption("No recent searches. Questions you ask will appear here.")

    st.markdown("---")


    tab_upload, tab_manage, tab_settings = st.tabs([
        ":material/upload_file: Upload",
        ":material/folder: Library",
        ":material/tune: Settings",
    ])

    # ---------- TAB 1: UPLOAD DOCUMENTS ----------
    with tab_upload:
        st.markdown("##### Upload PDF Documents")
        st.caption("Upload course PDFs. Vectors are automatically saved to persistent ChromaDB.")

        if getattr(rag_engine, "provider", "ollama") == "gemini":
            st.success("Connected to **Google Gemini AI** for cloud embeddings & inference.", icon=":material/cloud_done:")
        elif is_railway:
            st.warning(
                "**Railway Cloud Deployment**: Runs exclusively with Google Gemini. "
                "Add `GEMINI_API_KEY` to your Railway Variables or enter it in the **Settings** tab below.",
                icon=":material/cloud_upload:",
            )

        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or multiple PDF documents to ingest into the vector store.",
            label_visibility="collapsed",
        )

        if st.button(":material/bolt: Index New Documents", type="primary", width="stretch"):
            if not model_service_available:
                st.error("Google Gemini API Key is required to index documents on Railway. Please enter it in the Settings tab or above.", icon=":material/key:")
            elif not uploaded_files:
                st.warning("Please choose at least one PDF file to upload.", icon=":material/warning:")
            else:
                progress_bar = st.progress(0.0, text="Preparing documents...")
                total_new_chunks = 0
                skipped_files: list[str] = []
                indexed_files_current = st.session_state["indexed_files"]

                for i, u_file in enumerate(uploaded_files):
                    fn = u_file.name
                    if fn in indexed_files_current:
                        skipped_files.append(fn)
                        continue

                    file_bytes = u_file.read()

                    def _on_progress(processed: int, total: int, message: str) -> None:
                        pct = min(1.0, processed / max(1, total))
                        progress_bar.progress(pct, text=message)

                    try:
                        progress_bar.progress(0.0, text=f"Processing '{fn}'...")
                        chunk_count = rag_engine.ingest_pdf(
                            file_bytes,
                            fn,
                            progress_callback=_on_progress,
                        )
                        total_new_chunks += chunk_count
                    except DocumentProcessingError as e:
                        st.error(str(e), icon=":material/error:")
                    except Exception as e:
                        st.error(f"Error processing '{fn}': {e}", icon=":material/error:")

                progress_bar.empty()

                st.session_state["indexed_files"] = rag_engine.get_indexed_documents()

                if skipped_files:
                    st.info(
                        f"Skipped {len(skipped_files)} already-indexed file(s): {', '.join(skipped_files)}",
                        icon=":material/info:",
                    )
                if total_new_chunks > 0:
                    st.toast(
                        f"Indexed {len(uploaded_files) - len(skipped_files)} document(s) ({total_new_chunks} chunks).",
                        icon=":material/check_circle:",
                    )
                    st.rerun()

    # ---------- TAB 2: MANAGE & REMOVE DOCUMENTS ----------
    with tab_manage:
        doc_catalog = st.session_state["indexed_files"]
        doc_count = len(doc_catalog)

        st.markdown(f"##### Course Library ({doc_count})")
        st.caption("Documents below are preserved in storage across server restarts.")

        if not doc_catalog:
            st.info("No documents currently indexed. Upload PDFs in the Upload tab.", icon=":material/inbox:")
        else:
            search_query = ""
            if doc_count > 3:
                search_query = st.text_input(
                    "Filter documents...",
                    placeholder="Search document names...",
                    label_visibility="collapsed",
                ).strip().lower()

            filtered_docs = {
                k: v for k, v in doc_catalog.items()
                if search_query in k.lower()
            } if search_query else doc_catalog

            for filename, meta in list(filtered_docs.items()):
                with st.container(border=True):
                    st.markdown(f"**:material/picture_as_pdf: {filename}**")
                    chunk_c = meta.get("chunk_count", 0)
                    page_c = meta.get("page_count", 1)
                    st.markdown(f":blue-badge[{chunk_c} chunks] :gray-badge[{page_c} pages]")

                    if st.button(
                        ":material/delete: Remove document",
                        key=f"del_{filename}",
                        width="stretch",
                        help=f"Delete '{filename}' and all its vector chunks from the database.",
                    ):
                        try:
                            removed = rag_engine.delete_document(filename)
                            st.session_state["indexed_files"] = rag_engine.get_indexed_documents()
                            st.toast(f"Removed '{filename}' ({removed} chunks deleted).", icon=":material/delete:")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove document: {e}", icon=":material/error:")

            st.space("small")

            with st.popover(":material/delete_forever: Reset Knowledge Base", width="stretch"):
                st.markdown("⚠️ **Warning: Reset Entire Database**")
                st.write("Permanently delete all indexed PDF documents and vector embeddings from storage.")
                if st.button(":material/check: Confirm and Delete All", type="primary", width="stretch"):
                    cleared_chunks = rag_engine.clear()
                    st.session_state["indexed_files"] = {}
                    st.session_state["messages"] = []
                    st.session_state["thread_id"] = f"session_{uuid.uuid4().hex[:8]}"
                    st.toast(f"Database wiped ({cleared_chunks} chunks removed).", icon=":material/check_circle:")
                    st.rerun()

    # ---------- TAB 3: SETTINGS & RUNTIME ----------
    with tab_settings:
        st.markdown("##### Session & Runtime")
        st.caption(f"Session Thread: `{st.session_state['thread_id']}`")

        if st.button(":material/restart_alt: Clear Chat History", width="stretch"):
            st.session_state["messages"] = []
            st.session_state["thread_id"] = f"session_{uuid.uuid4().hex[:8]}"
            st.toast("Chat history cleared. Course documents remain intact.", icon=":material/refresh:")
            st.rerun()

        st.space("small")
        st.markdown("##### Provider Configuration")
        current_gemini = os.getenv("GEMINI_API_KEY", "")
        gemini_input = st.text_input(
            "Google Gemini API Key",
            value=current_gemini,
            type="password",
            placeholder="AIzaSy...",
            help="Provide your Google Gemini API key to enable cloud embeddings (text-embedding-004) and chat (gemini-2.0-flash).",
        )
        if st.button(":material/save: Save & Apply Provider", width="stretch"):
            if gemini_input.strip():
                os.environ["GEMINI_API_KEY"] = gemini_input.strip()
            else:
                os.environ.pop("GEMINI_API_KEY", None)
            st.cache_resource.clear()
            st.toast("Provider configuration updated! Reloading...", icon=":material/check_circle:")
            st.rerun()

        st.space("small")
        st.markdown("##### Infrastructure Specs")
        provider_name = "Google Gemini AI" if getattr(rag_engine, "provider", "ollama") == "gemini" else "Local Ollama"
        st.caption(
            f":material/cloud: **Provider:** `{provider_name}`\n\n"
            f":material/smart_toy: **LLM Model:** `{agent_service.model_name}`\n\n"
            f":material/search: **Embedding Model:** `{rag_engine.embed_model}`\n\n"
            f":material/folder_open: **Storage Path:** `{rag_engine.persist_directory}`\n\n"
            f":material/speed: **Confidence Threshold:** `{rag_engine.relevance_threshold}`"
        )


# ================= MAIN AREA: GOOGLE GEMINI STYLE CANVAS =================

active_docs = len(st.session_state["indexed_files"])
total_chunks = rag_engine.document_count

# Gemini Minimal Top Navigation Bar
st.markdown(
    f"""<div class="gemini-nav">
        <div class="gemini-logo">
            <span class="gemini-sparkle">✦</span>
            <span>Professor AI</span>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="gemini-badge gemini-badge-model">{agent_service.model_name}</span>
            <span class="gemini-badge gemini-badge-docs">{active_docs} PDF(s)</span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

if not model_service_available:
    banner_title = "Google Gemini API Key Required on Railway" if is_railway else "Local Ollama Model is Not Available"
    banner_desc = (
        "Railway runs 100% in the cloud and uses Google Gemini exclusively. "
        "Please connect your <b>Google Gemini API Key</b> below or add <code>GEMINI_API_KEY</code> to your Railway Variables."
        if is_railway else
        "Professor AI cannot connect to a local Ollama server at <code>http://localhost:11434</code>. "
        "To activate cloud inference without running Ollama locally, please connect your <b>Google Gemini API Key</b> below."
    )
    st.markdown(
        f"""<div style="background: rgba(234, 67, 53, 0.08); border: 1px solid rgba(234, 67, 53, 0.25); border-radius: 16px; padding: 20px 24px; margin: 16px 0 20px 0;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="font-size: 22px;">⚡</span>
                <span style="font-size: 16px; font-weight: 600; color: #f28b82;">{banner_title}</span>
            </div>
            <p style="color: #c4c7c5; font-size: 13.5px; margin: 0; line-height: 1.5;">
                {banner_desc}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("##### :material/key: Connect Google Gemini Cloud Agent")
        st.caption("Get a free API key in seconds at [Google AI Studio](https://aistudio.google.com/app/apikey).")
        k_col, b_col, l_col = st.columns([3, 1.5, 1.5])
        with k_col:
            main_gemini_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Paste AIzaSy... key here",
                label_visibility="collapsed",
                key="main_gemini_key_input",
            )
        with b_col:
            if st.button(":material/bolt: Activate Gemini", type="primary", width="stretch"):
                if main_gemini_input.strip():
                    os.environ["GEMINI_API_KEY"] = main_gemini_input.strip()
                    st.cache_resource.clear()
                    st.toast("Google Gemini Agent activated successfully!", icon=":material/check_circle:")
                    st.rerun()
                else:
                    st.warning("Please paste a valid key.")
        with l_col:
            st.link_button(
                ":material/open_in_new: Get Free Key",
                "https://aistudio.google.com/app/apikey",
                width="stretch",
            )

# 1. Gemini Welcome State (only visible when there is no active conversation)
if not has_conversation and active_docs > 0:
    st.markdown(
        """<div class="gemini-hero">
            <div class="gemini-hero-title">Hello, Student</div>
            <div class="gemini-hero-sub">How can I help you master your course materials today?</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Gemini horizontal suggestion chips
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("📐 Teach me a core concept\nwith visual Mermaid diagram", width="stretch"):
            st.session_state["pending_prompt"] = "Professor, please choose a core concept from the uploaded course materials, explain the intuition clearly, and provide a visual Mermaid diagram illustrating how it works."
            st.rerun()
    with p_col2:
        if st.button("🔬 Solve an exam problem\nstep-by-step with formulas", width="stretch"):
            st.session_state["pending_prompt"] = "Professor, pick an important problem or calculation from the course materials. State the problem, governing formulas in LaTeX, and walk me through the step-by-step solution with a visual workflow diagram."
            st.rerun()
    with p_col3:
        if st.button("📝 5 key practice questions\nfor exam preparation", width="stretch"):
            st.session_state["pending_prompt"] = "Professor, based on the uploaded course PDFs, what are 5 key practice questions or exam problems students should prepare for?"
            st.rerun()

elif not has_conversation and active_docs == 0:
    st.markdown(
        """<div class="gemini-hero">
            <div class="gemini-hero-title">Welcome to Classroom</div>
            <div class="gemini-hero-sub">Upload your course PDFs via the left sidebar to begin studying.</div>
        </div>""",
        unsafe_allow_html=True,
    )

# 3. Render Historical Chat Stream (Gemini style)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(f"**{msg['content']}**")
    else:
        with st.chat_message("assistant", avatar="✨"):
            _render_gemini_response(msg["content"], msg.get("citations", []))

# 4. Process New User Query
if user_query:
    # 1. Log and render user message
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(f"**{user_query}**")

    # 2. Assistant execution with live status and thinking disclosure
    with st.chat_message("assistant", avatar="✨"):
        with st.status(":shimmer[Searching course documents & preparing lesson...]", type="compact") as status:
            result = agent_service.ask(
                user_question=user_query,
                thread_id=st.session_state["thread_id"],
            )
            citations = result.get("citations", [])
            if citations:
                status.update(label=f"Verified with {len(citations)} course excerpt(s)", state="complete")
            else:
                status.update(label="Course material search completed", state="complete")

        answer = result["answer"]
        _render_gemini_response(answer, citations)

        # 3. Save assistant message with citations to history
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
            }
        )


