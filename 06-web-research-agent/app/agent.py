from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.models import Citation, ToolTrace
from app.tools import FetchTool, SearchTool


SYSTEM_PROMPT = """You are a production web research agent.

Your job is to answer user questions accurately using the provided tools:
- search_web: find relevant and current pages
- fetch_webpage: read and extract content from a specific page

Rules:
1. Use search_web when the question needs current or web-backed information.
2. Use fetch_webpage to inspect promising pages before making factual claims.
3. Base claims on fetched content whenever possible.
4. Be concise but complete.
5. Cite the URLs you actually used in the answer.
6. If a required tool is unavailable, say that clearly.
"""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "search_web",
        "description": "Search the web with Serper and return a small list of relevant URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "fetch_webpage",
        "description": "Fetch and extract the main content from a web page using Crawl4AI.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


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
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def answer(self, question: str, history: list[dict[str, Any]]) -> AgentResult:
        client = self._get_client()
        input_items = [
            {"role": message["role"], "content": message["content"]}
            for message in history
        ]
        input_items.append({"role": "user", "content": question})

        response = await client.responses.create(
            model=settings.llm_model,
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=TOOL_SCHEMAS,
        )

        tool_traces: list[ToolTrace] = []
        citations_by_url: dict[str, Citation] = {}
        input_tokens = 0
        output_tokens = 0
        tool_cost_usd = 0.0

        for _ in range(settings.max_tool_rounds):
            usage = getattr(response, "usage", None)
            if usage:
                input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                break

            tool_outputs = []
            for call in function_calls:
                arguments = json.loads(call.arguments)
                result, trace, cost_increment = await self._run_tool(call.name, arguments)
                tool_traces.append(trace)
                tool_cost_usd += cost_increment
                self._capture_citations(call.name, result, citations_by_url)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=True),
                    }
                )

            response = await client.responses.create(
                model=settings.llm_model,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=TOOL_SCHEMAS,
            )

        final_usage = getattr(response, "usage", None)
        if final_usage:
            input_tokens += int(getattr(final_usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(final_usage, "output_tokens", 0) or 0)

        answer_text = getattr(response, "output_text", "") or ""
        if not answer_text:
            answer_text = "I could not generate an answer."

        return AgentResult(
            answer=answer_text.strip(),
            citations=list(citations_by_url.values()),
            tool_traces=tool_traces,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_cost_usd=tool_cost_usd,
        )

    async def _run_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], ToolTrace, float]:
        try:
            if tool_name == self._search_tool.name:
                result = self._search_tool.run(
                    query=arguments["query"],
                    limit=arguments["limit"],
                )
                trace = ToolTrace(
                    name=tool_name,
                    input=arguments,
                    success=True,
                    summary=f"Found {len(result['results'])} search results.",
                )
                return result, trace, settings.search_tool_cost_usd

            if tool_name == self._fetch_tool.name:
                result = await self._fetch_tool.run(url=arguments["url"])
                trace = ToolTrace(
                    name=tool_name,
                    input=arguments,
                    success=True,
                    summary=f"Fetched {result['markdown_chars']} characters from page.",
                )
                return result, trace, settings.fetch_tool_cost_usd

            raise RuntimeError(f"Unknown tool: {tool_name}")
        except Exception as exc:  # noqa: BLE001
            result = {
                "error": str(exc),
                "input": arguments,
            }
            trace = ToolTrace(
                name=tool_name,
                input=arguments,
                success=False,
                summary=str(exc),
            )
            return result, trace, 0.0

    @staticmethod
    def _capture_citations(tool_name: str, result: dict[str, Any], citations_by_url: dict[str, Citation]) -> None:
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
