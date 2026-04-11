import json

from .base import BaseCollector


class ArbeitnowCollector(BaseCollector):
    URL = "https://www.arbeitnow.com/api/job-board-api"

    def __init__(self, run_budget: int = 5):
        super().__init__("arbeitnow", run_budget)

    def fetch_page(self, query: str, page: int) -> list:
        resp = self._fetch_with_backoff(self.URL, params={"page": page})
        data = resp.json().get("data", [])
        return [self._normalize_job(j) for j in data]

    def _normalize_job(self, item: dict) -> dict:
        return {
            "external_id": item.get("slug"),
            "source_system": "arbeitnow",
            "source_board": None,
            "title": item.get("title"),
            "location": item.get("location"),
            "description": item.get("description"),
            "company_name": item.get("company_name"),
            "remote_status": "remote" if item.get("remote") else "onsite",
            "posted_date": item.get("created_at"),
            "source_url": item.get("url"),
            "apply_url": item.get("url"),
            "raw_data": json.dumps(item),
            "scraped_at": None,
        }
