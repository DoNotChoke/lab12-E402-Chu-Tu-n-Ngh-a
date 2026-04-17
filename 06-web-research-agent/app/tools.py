from __future__ import annotations

from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from langchain_community.utilities import GoogleSerperAPIWrapper

from app.config import settings


class SearchTool:
    name = "search_web"

    def is_configured(self) -> bool:
        return bool(settings.serper_api_key)

    def run(self, query: str, limit: int | None = None) -> dict:
        if not self.is_configured():
            raise RuntimeError("SERPER_API_KEY is not configured.")

        effective_limit = max(1, min(limit or settings.search_results_limit, 10))
        wrapper = GoogleSerperAPIWrapper(
            serper_api_key=settings.serper_api_key,
            k=effective_limit,
        )
        raw = wrapper.results(query)
        normalized_results = []

        for bucket in ("organic", "news", "places"):
            for item in raw.get(bucket, []) or []:
                url = item.get("link") or item.get("url")
                if not url:
                    continue
                normalized_results.append(
                    {
                        "title": item.get("title") or url,
                        "url": url,
                        "snippet": item.get("snippet") or item.get("description") or "",
                        "source_type": bucket,
                    }
                )
                if len(normalized_results) >= effective_limit:
                    break
            if len(normalized_results) >= effective_limit:
                break

        return {
            "query": query,
            "results": normalized_results,
        }


class FetchTool:
    name = "fetch_webpage"

    async def run(self, url: str) -> dict:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http and https URLs are supported.")

        browser_config = BrowserConfig(headless=True, verbose=False)
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=settings.crawler_page_timeout_ms,
            check_robots_txt=True,
            word_count_threshold=10,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawl_config)

        success = getattr(result, "success", True)
        if not success:
            raise RuntimeError(getattr(result, "error_message", "crawl failed"))

        metadata = getattr(result, "metadata", {}) or {}
        markdown = self._extract_markdown(result)
        trimmed_markdown = markdown[: settings.fetch_max_markdown_chars]

        return {
            "url": url,
            "title": metadata.get("title") if isinstance(metadata, dict) else None,
            "markdown": trimmed_markdown,
            "markdown_chars": len(trimmed_markdown),
        }

    @staticmethod
    def _extract_markdown(result: object) -> str:
        markdown_v2 = getattr(result, "markdown_v2", None)
        if markdown_v2 and getattr(markdown_v2, "raw_markdown", None):
            return markdown_v2.raw_markdown

        markdown = getattr(result, "markdown", None)
        if markdown:
            return markdown

        cleaned_html = getattr(result, "cleaned_html", None)
        if cleaned_html:
            return cleaned_html

        return ""
