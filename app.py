import streamlit as st
import os
from graph import build_graph

# =========================================================
#   CONFIGURACIÓN DE LA APP
# =========================================================
st.set_page_config(
    page_title="Business Research Assistant",
    page_icon="🔎",
    layout="wide"
)

# =========================================================
#   SIDEBAR PROFESIONAL
# =========================================================
with st.sidebar:
    st.title("🔎 Asistente de Investigación BI")
    st.markdown("""
    Esta herramienta utiliza agentes autónomos:

    - 🌍 Serper Search  
    - 🕸️ Tavily Scraping (profundo)  
    - 📚 Wikipedia  
    - 🧠 Gemini 2.5 Flash  
    - 🔀 LangGraph (Agente orquestador)

    Ingresas una pregunta → la IA investiga → analiza → entrega insights empresariales.
    """)
    st.divider()
    st.caption("Power by Oscar Carabali + Gemini 2.5 Flash")

# =========================================================
#   INTERFAZ PRINCIPAL
# =========================================================
st.title("🔎 Asistente de Investigación Empresarial")
st.write("Haz una pregunta, por ejemplo: **“Analiza los competidores del grupo Argos”**")

user_question = st.text_input("Tu pregunta:")

# Cargar grafo
graph = build_graph()

# Exportar Mermaid
mermaid_graph = graph.get_graph().draw_mermaid()

# Mostrar Mermaid en expander
with st.expander("📊 Ver grafo LangGraph (Mermaid)"):

    mermaid_html = f"""
    <div class="mermaid">
    {mermaid_graph}
    </div>

    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: true }});
    </script>
    """

    st.components.v1.html(mermaid_html, height=500, scrolling=True)

# =========================================================
#   EJECUCIÓN DEL GRAFO
# =========================================================
if st.button("Ejecutar análisis"):

    if not user_question:
        st.warning("Por favor ingresa una pregunta.")
        st.stop()

    with st.spinner("🔍 Recolectando información, analizando y generando insights…"):

        # Ejecutar el grafo completo
        result = graph.invoke({"question": user_question})

        final_answer = result.get("final", "No se pudo generar respuesta.")

    st.subheader("📌 Resultado del análisis")
    st.markdown(final_answer)
