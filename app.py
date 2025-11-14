import operator
import os
from dotenv import load_dotenv
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI  # << GEMINI

from langgraph.graph import StateGraph, START, END

# ================================
# 1. Configuración inicial
# ================================

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


# ================================
# 2. Nodos
# ================================

def search_web(state):

    tavily_key = os.getenv("TAVILY_API_KEY")

    tavily_search = TavilySearch(
        max_results=3,
        tavily_api_key=tavily_key
    )

    search_docs = tavily_search.invoke(state["question"])
    results = search_docs.get("results", [])

    formatted = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in results
    ])

    return {"context": [formatted]}


def search_wikipedia(state):

    search_docs = WikipediaLoader(
        query=state["question"],
        load_max_docs=2
    ).load()

    formatted = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page","")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_docs
    ])

    return {"context": [formatted]}


def generate_answer(state):

    context = state["context"]
    question = state["question"]

    template = f"""
Responde de forma clara y directa usando únicamente el siguiente contexto:

Pregunta:
{question}

Contexto:
{context}
"""

    answer = llm.invoke([
        SystemMessage(content=template),
        HumanMessage(content="Responde la pregunta.")
    ])

    return {"answer": answer}


# ================================
# 3. Construcción del Grafo
# ================================

builder = StateGraph(State)

builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")

builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")

builder.add_edge("generate_answer", END)

graph = builder.compile()


# ================================
# 4. Interfaz en Streamlit
# ================================

st.title("🔎 Agente Web + Wikipedia con Gemini 2.0-Flash")
st.write("Haz una pregunta y la IA buscará información en la web y en Wikipedia.")

question = st.text_input("Escribe tu pregunta:")

if st.button("Preguntar"):

    if not question:
        st.warning("Por favor ingresa una pregunta.")
    else:
        with st.spinner("Buscando información…"):

            result = graph.invoke({"question": question})
            final_answer = result["answer"].content

        st.subheader("❓ Tu pregunta")
        st.write(question)

        st.subheader("💡 Respuesta del agente")
        st.write(final_answer)
