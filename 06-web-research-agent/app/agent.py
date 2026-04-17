from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.models import Citation, ToolTrace
from app.tools import FetchTool, SearchTool


SYSTEM_PROMPT = """You are a production web research agent.

Your job is to answer user questions accurately using the provided tools:
- search_web: find relevant and current pages
- fetch_webpage: read and extract content from a specific page

Rules:
1. Use search_web when the question needs current or web-backed information.
2. After search, fetch the most relevant pages before making factual claims.
3. Use at most 3 fetch_webpage calls unless the user explicitly needs more depth.
4. Base claims on fetched content whenever possible.
5. Return a direct final answer once you have enough information.
6. Cite the URLs you actually used.
7. Do not keep calling tools once you already have enough evidence.
"""


@dataclass
class AgentResult:
    answer: str
    citations: list[Citation]
    tool_traces: list[ToolTrace]
    input_tokens: int
    output_tokens: int
    tool_cost_usd: float


class ResearchAgent:
    def __init__(self) -> None:
        self._search_tool = SearchTool()
        self._fetch_tool = FetchTool()
        self._model: ChatOpenAI | None = None

    def _get_model(self) -> ChatOpenAI:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        if self._model is None:
            self._model = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.openai_api_key,
                stream_usage=True,
                use_responses_api=True,
                reasoning_effort="medium",
            )
        return self._model

    async def answer(self, question: str, history: list[dict[str, Any]]) -> AgentResult:
        tool_traces: list[ToolTrace] = []
        citations_by_url: dict[str, Citation] = {}
        tool_cost_usd = 0.0

        @tool
        def search_web(query: str, limit: int = 5) -> str:
            """Search the web for relevant URLs and snippets."""
            nonlocal tool_cost_usd
            try:
                result = self._search_tool.run(query=query, limit=limit)
                tool_traces.append(
                    ToolTrace(
                        name="search_web",
                        input={"query": query, "limit": limit},
                        success=True,
                        summary=f"Found {len(result['results'])} search results.",
                    )
                )
                self._capture_citations("search_web", result, citations_by_url)
                tool_cost_usd += settings.search_tool_cost_usd
                return json.dumps(result, ensure_ascii=True)
            except Exception as exc:  # noqa: BLE001
                tool_traces.append(
                    ToolTrace(
                        name="search_web",
                        input={"query": query, "limit": limit},
                        success=False,
                        summary=str(exc),
                    )
                )
                return json.dumps({"error": str(exc), "query": query, "limit": limit}, ensure_ascii=True)

        @tool
        async def fetch_webpage(url: str) -> str:
            """Fetch a webpage and extract the main content as markdown."""
            nonlocal tool_cost_usd
            try:
                result = await self._fetch_tool.run(url=url)
                tool_traces.append(
                    ToolTrace(
                        name="fetch_webpage",
                        input={"url": url},
                        success=True,
                        summary=f"Fetched {result['markdown_chars']} characters from page.",
                    )
                )
                self._capture_citations("fetch_webpage", result, citations_by_url)
                tool_cost_usd += settings.fetch_tool_cost_usd
                return json.dumps(result, ensure_ascii=True)
            except Exception as exc:  # noqa: BLE001
                tool_traces.append(
                    ToolTrace(
                        name="fetch_webpage",
                        input={"url": url},
                        success=False,
                        summary=str(exc),
                    )
                )
                return json.dumps({"error": str(exc), "url": url}, ensure_ascii=True)

        agent = create_agent(
            model=self._get_model(),
            tools=[search_web, fetch_webpage],
            system_prompt=SYSTEM_PROMPT,
        )

        input_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in history
        ]
        input_messages.append({"role": "user", "content": question})

        result = await agent.ainvoke({"messages": input_messages})
        messages = result.get("messages", [])
        answer = self._extract_final_answer(messages)
        input_tokens, output_tokens = self._extract_usage(messages)

        return AgentResult(
            answer=answer,
            citations=list(citations_by_url.values()),
            tool_traces=tool_traces,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_cost_usd=tool_cost_usd,
        )

    @staticmethod
    def _extract_final_answer(messages: list[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = ResearchAgent._message_text(message)
                if text:
                    return text.strip()
        return "I could not generate an answer."

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if not isinstance(item, dict):
                    continue

                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    chunks.append(str(item["text"]))
                    continue

                if item.get("type") == "reasoning" and item.get("summary"):
                    summary = item["summary"]
                    if isinstance(summary, list):
                        for part in summary:
                            if isinstance(part, dict) and part.get("text"):
                                chunks.append(str(part["text"]))

            return "\n".join(chunk for chunk in chunks if chunk).strip()

        return ""

    @staticmethod
    def _extract_usage(messages: list[Any]) -> tuple[int, int]:
        input_tokens = 0
        output_tokens = 0

        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            usage = getattr(message, "usage_metadata", None) or {}
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)

        return input_tokens, output_tokens

    @staticmethod
    def _capture_citations(
        tool_name: str,
        result: dict[str, Any],
        citations_by_url: dict[str, Citation],
    ) -> None:
        if tool_name == "search_web":
            for item in result.get("results", []):
                url = item.get("url")
                if url and url not in citations_by_url:
                    citations_by_url[url] = Citation(
                        title=item.get("title"),
                        url=url,
                        source_type=item.get("source_type", "search"),
                    )

        if tool_name == "fetch_webpage":
            url = result.get("url")
            if url and url not in citations_by_url:
                citations_by_url[url] = Citation(
                    title=result.get("title"),
                    url=url,
                    source_type="fetch",
                )
