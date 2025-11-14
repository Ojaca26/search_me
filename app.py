import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilyClient

# =========================================================
#       WIKIPEDIA LOADER
# =========================================================

def tool_wikipedia(query):
    try:
        docs = WikipediaLoader(query=query, load_max_docs=2).load()
        formatted = ""

        for d in docs:
            formatted += f"\n\n[TÍTULO: {d.metadata.get('title','')}]"
            formatted += f"\n{d.page_content}\n"
        return formatted

    except Exception as e:
        return f"[Wikipedia Error] {e}"


# =========================================================
#       SERPER SEARCH
# =========================================================
def tool_serper_search(query):
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    url = "https://google.serper.dev/search"

    payload = {
        "q": query,
        "gl": "co",
        "hl": "es"
    }

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(url, json=payload, headers=headers).json()
        results = res.get("organic", [])

        formatted = ""
        for r in results[:7]:
            formatted += f"\n\n### {r.get('title')}\n{r.get('snippet')}\n{r.get('link')}"
        return formatted

    except Exception as e:
        return f"[Serper Error] {e}"


# =========================================================
#       TAVILY SEARCH (Scraping profundo)
# =========================================================
def tool_tavily_scrape(query):
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        result = tavily.search(
            query=query,
            include_raw_content=True,
            include_domains=["*"],
            max_results=7
        )

        formatted = ""
        for i in result.get("results", []):
            formatted += f"\n\n## {i['title']}\n"
            formatted += f"{i.get('raw_content','')[:2000]}"
            formatted += f"\nURL: {i['url']}\n"

        return formatted

    except Exception as e:
        return f"[Tavily Error] {e}"
