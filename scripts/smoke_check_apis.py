#!/usr/bin/env python3
"""
Light smoke test for job API collectors.
Tests jsearch, serpapi, jobsearch, jobsapi with 1 request each.
Run from repo root: python scripts/test_apis.py
"""
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load .env
load_dotenv()

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.collectors.jsearch import JSearchCollector
from data_pipeline.collectors.serpapi import SerpApiCollector
from data_pipeline.collectors.jobsearch import JobSearchCollector
from data_pipeline.collectors.jobsapi import JobsApiCollector


def test_collector(collector_class, name: str) -> None:
    """Test a single collector with 1 request."""
    try:
        collector = collector_class(run_budget=1)
        jobs = collector.fetch_page("AI engineer", page=1)

        if not jobs:
            print(f"[{name:8}] PASS   0 jobs")
            return

        first = jobs[0]
        title = first.get("title") or first.get("job_title") or "?"
        company = first.get("company_name") or first.get("employer_name") or "?"

        print(f"[{name:8}] PASS   {len(jobs):2} jobs  | \"{title}\" @ {company}")
    except Exception as e:
        print(f"[{name:8}] FAIL   {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    print("Testing API collectors (1 request each)...\n")
    test_collector(JSearchCollector, "jsearch")
    test_collector(SerpApiCollector, "serpapi")
    test_collector(JobSearchCollector, "jobsearch")
    test_collector(JobsApiCollector, "jobsapi")
    print()
