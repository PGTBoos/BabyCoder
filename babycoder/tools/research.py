"""
babycoder.tools.research

AGENT RESEARCH toolkit: read-only outbound HTTP. The only network path
besides the model itself, so withholding it is what keeps an agent offline.
Both backends are optional; a missing package returns NOTFOUND, not a crash.
"""

import requests

from ..core import P, tool

# Wikimedia blocks generic User-Agents far sooner than it blocks volume.
# Put real contact details in here.
USER_AGENT = "babycoder-agent/1.0 (https://github.com/PGTBoos/BabyCoder; local research agent)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"


@tool("Search Wikipedia and read article summaries.",
      query=P("string"), max_results=P("integer", optional=True))
def search_wikipedia(query, max_results=2):
    headers = {"User-Agent": USER_AGENT}
    try:
        hits = requests.get(WIKIPEDIA_API, timeout=20, headers=headers, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": max_results, "format": "json",
        }).json()["query"]["search"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        return f"NOTFOUND: could not reach Wikipedia ({exc})"
    out = []
    for hit in hits[:max_results]:
        title = hit.get("title", "")
        try:
            page = requests.get(WIKIPEDIA_SUMMARY + title.replace(" ", "_"), timeout=20, headers=headers).json()
        except (requests.RequestException, ValueError):
            continue
        if page.get("extract"):
            url = page.get("content_urls", {}).get("desktop", {}).get("page", "")
            out.append(f"## {title}\n{page['extract'][:600]}\n({url})")
    return "\n\n".join(out) if out else f"NOTFOUND: nothing usable on Wikipedia for {query!r}"


def _ddgs_class():
    """The DuckDuckGo client under either its new (ddgs) or old name."""
    for module in ("ddgs", "duckduckgo_search"):
        try:
            return __import__(module).DDGS
        except (ImportError, AttributeError):
            continue
    return None


@tool("Search the web and read result snippets. Read-only.",
      query=P("string"), max_results=P("integer", optional=True))
def search_web(query, max_results=3):
    ddgs = _ddgs_class()
    if ddgs is None:
        return "NOTFOUND: web search needs the ddgs package. Install it with: pip install ddgs"
    try:
        results = list(ddgs().text(query, max_results=max_results))
    except Exception as exc:
        return f"NOTFOUND: web search failed ({exc})"
    out = [f"## {r.get('title', '')}\n{(r.get('body') or '')[:300]}\n({r.get('href', '')})" for r in results]
    return "\n\n".join(out) if out else f"NOTFOUND: no web results for {query!r}"


@tool("Look a topic up in Wikipedia and the web at once.", topic=P("string"))
def research_topic(topic):
    parts, failures = [], []
    for label, text in (("Wikipedia", search_wikipedia(topic, 2)), ("Web", search_web(topic, 2))):
        if text.startswith("NOTFOUND:"):
            failures.append(f"{label}: {text[len('NOTFOUND: '):]}")
        else:
            parts.append(f"# {label} on {topic}\n{text}")
    if not parts:
        return f"NOTFOUND: found nothing on {topic!r}. " + "; ".join(failures)
    return "\n\n".join(parts)
