from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tools import tool_wikipedia, tool_serper_search, tool_tavily_scrape

# =============================================================
#        1. CLASSIFIER AGENT
# =============================================================
def classifier_agent(question):

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = f"""
        Clasifica la siguiente pregunta según la intención:

        Pregunta:
        {question}

        Categorías posibles:
        - competidores
        - mercado
        - industria
        - reputación
        - análisis general
        - análisis financiero
        - estrategia
        - tendencias
        - mix
        - otro

        Devuelve SOLO el nombre de la categoría en minúsculas.
    """

    res = llm.invoke([HumanMessage(content=prompt)])
    return res.content.strip().lower()


# =============================================================
#        2. RESEARCH AGENT
# =============================================================
def research_agent(question, category):

    context = ""

    # Serper siempre
    context += "\n\n### SERPER SEARCH:\n"
    context += tool_serper_search(question)

    # Tavily depende del caso
    context += "\n\n### TAVILY SCRAPING:\n"
    context += tool_tavily_scrape(question)

    # Wikipedia a demanda
    if category in ["industria", "competidores", "mercado"]:
        context += "\n\n### WIKIPEDIA:\n"
        context += tool_wikipedia(question)

    return context


# =============================================================
#        3. ANALYZER AGENT (Gemini 2.5 Flash)
# =============================================================
def analyzer_agent(question, context):

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt = f"""
    Eres un analista empresarial senior en Business Intelligence.
    Analiza la pregunta y la información recolectada.

    Pregunta:
    {question}

    Información recopilada de múltiples fuentes:
    {context}

    Produce el siguiente output:

    ## Resumen Ejecutivo
    (5–8 líneas)

    ## Análisis Empresarial
    (Tendencias, riesgos, oportunidades)

    ## Insights Clave
    - Insight 1
    - Insight 2
    - Insight 3

    ## Competidores / Benchmark (si aplica)

    ## Recomendaciones Estratégicas

    ## Fuentes
    Lista las URLs encontradas.
    """

    res = llm.invoke([HumanMessage(content=prompt)])
    return res.content


# =============================================================
#        4. COMPOSER AGENT
# =============================================================
def composer_agent(result):
    return f"""
    # 🔎 Informe de Investigación Empresarial

    {result}

    ---
    _Asistente IA BI — Powered by DataInsights + Gemini 2.5 Flash_
    """
