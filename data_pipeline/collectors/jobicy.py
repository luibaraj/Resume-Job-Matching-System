import json
import time

from .base import BaseCollector


class JobicyCollector(BaseCollector):
    URL = "https://jobicy.com/api/v2/remote-jobs"

    def __init__(self, run_budget: int = 10):
        super().__init__("jobicy", run_budget)

    def fetch_page(self, query: str, page: int) -> list:
        if page > 1:
            return []  # no pagination support
        time.sleep(1)
        params = {"count": 50, "tag": query}
        resp = self._fetch_with_backoff(self.URL, params=params)
        return [self._normalize_job(j) for j in resp.json().get("jobs", [])]

    def _normalize_job(self, item: dict) -> dict:
        return {
            "external_id": str(item.get("id")),
            "source_system": "jobicy",
            "source_board": None,
            "title": item.get("jobTitle"),
            "location": item.get("jobGeo"),
            "description": item.get("jobDescription"),
            "company_name": item.get("companyName"),
            "employment_type": (item.get("jobType") or [None])[0],
            "remote_status": "remote",
            "posted_date": item.get("pubDate"),
            "source_url": item.get("url"),
            "apply_url": item.get("url"),
            "salary_min": item.get("salaryMin"),
            "salary_max": item.get("salaryMax"),
            "salary_currency": item.get("salaryCurrency"),
            "salary_period": item.get("salaryPeriod"),
            "raw_data": json.dumps(item),
            "scraped_at": None,
        }
