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

# Inicializar Grafo
graph = build_graph()
mermaid_graph = graph.get_graph().draw_mermaid()

# =========================================================
#   SIDEBAR PROFESIONAL
# =========================================================
with st.sidebar:
    st.caption("Power by Oscar Carabali + Gemini 2.5 Flash")
    st.divider()
    
    st.title("🔎 Asistente de Investigación BI")
    st.markdown("""
    Esta herramienta utiliza agentes autónomos:

    - 🌍 Serper Search  
    - 🕸️ Tavily Scraping (profundo)  
    - 📚 Wikipedia  
    - 🧠 Gemini 2.5 Flash  
    - 🔀 LangGraph (Agente orquestador)
    """)
    
    # Mostrar Grafo en Sidebar
    st.subheader("📊 Flujo de Agentes")
    mermaid_html = f"""
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
      <style>
        body {{ margin: 0; background: transparent; overflow: hidden; padding-bottom: 50px; }}
        #container {{ width: 100%; display: flex; justify-content: center; }}
        .mermaid svg {{ 
          width: 100% !important; 
          height: auto !important;
          background: white;
          border-radius: 8px;
          padding: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
      </style>
    </head>
    <body>
      <div id="container">
        <div class="mermaid">
{mermaid_graph}
        </div>
      </div>
      <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            htmlLabels: true,
            flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(mermaid_html, height=650)

# =========================================================
#   INTERFAZ PRINCIPAL
# =========================================================
st.title("🔎 Asistente de Investigación Empresarial")
st.write("Haz una pregunta, por ejemplo: **“Analiza los competidores del grupo Argos”**")

user_question = st.text_input("Tu pregunta:")

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
