from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

from agents import classifier_agent, research_agent, analyzer_agent, composer_agent


class State(TypedDict):
    question: str
    category: str
    context: Annotated[str, operator.add]
    analysis: str
    final: str


# NODOS
def node_classifier(state):
    cat = classifier_agent(state["question"])
    return {"category": cat}


def node_research(state):
    ctx = research_agent(state["question"], state["category"])
    return {"context": ctx}


def node_analyze(state):
    analysis = analyzer_agent(state["question"], state["context"])
    return {"analysis": analysis}


def node_compose(state):
    final = composer_agent(state["analysis"])
    return {"final": final}


# GRAFO
def build_graph():

    g = StateGraph(State)

    g.add_node("classifier", node_classifier)
    g.add_node("research", node_research)
    g.add_node("analyze", node_analyze)
    g.add_node("compose", node_compose)

    g.add_edge(START, "classifier")
    g.add_edge("classifier", "research")
    g.add_edge("research", "analyze")
    g.add_edge("analyze", "compose")
    g.add_edge("compose", END)

    return g.compile()
