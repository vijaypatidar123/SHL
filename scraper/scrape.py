"""
Scrape the SHL product catalog (Individual Test Solutions only).

Output: data/catalog.json — a list of assessment records with
  name, url, slug, test_types, remote_testing, adaptive_irt,
  description, job_levels, languages, duration_text,
  duration_minutes, untimed.

Usage: python -m scraper.scrape
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_FILE = DATA_DIR / "catalog.json"

BASE = "https://www.shl.com"
LISTING_URL = BASE + "/products/product-catalog/"
PAGE_SIZE = 12
INDIVIDUAL_HEADER = "Individual Test Solutions"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


@dataclass
class Assessment:
    name: str
    url: str
    slug: str
    test_types: list[str] = field(default_factory=list)
    test_type_labels: list[str] = field(default_factory=list)
    remote_testing: bool = False
    adaptive_irt: bool = False
    description: str = ""
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    duration_text: str = ""
    duration_minutes: int | None = None
    untimed: bool = False


def _has_yes(td: Tag) -> bool:
    span = td.find("span", class_="catalogue__circle")
    return bool(span and "-yes" in span.get("class", []))


def parse_listing_rows(html: str) -> list[dict]:
    """Extract minimal fields (name, url, test_types, flags) from one listing page."""
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for table in soup.find_all("table"):
        ths = table.find_all("th")
        if not ths or ths[0].get_text(strip=True) != INDIVIDUAL_HEADER:
            continue
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) != 4:
                continue
            link = tds[0].find("a")
            if not link or not link.get("href"):
                continue
            href = link["href"]
            url = href if href.startswith("http") else BASE + href
            slug_match = re.search(r"/view/([^/]+)/?", href)
            slug = slug_match.group(1) if slug_match else ""
            keys = [
                k.get_text(strip=True)
                for k in tds[3].find_all("span", class_="product-catalogue__key")
            ]
            out.append(
                {
                    "name": link.get_text(strip=True),
                    "url": url,
                    "slug": slug,
                    "remote_testing": _has_yes(tds[1]),
                    "adaptive_irt": _has_yes(tds[2]),
                    "test_types": keys,
                }
            )
    return out


def parse_detail(html: str) -> dict:
    """Extract description, job_levels, languages, duration from a detail page."""
    soup = BeautifulSoup(html, "lxml")
    fields: dict[str, list[str]] = {}
    for row in soup.find_all("div", class_="product-catalogue-training-calendar__row"):
        h = row.find(["h2", "h3", "h4"])
        if not h:
            continue
        heading = h.get_text(strip=True)
        paras = [p.get_text(" ", strip=True) for p in row.find_all("p")]
        fields[heading] = paras

    description = (fields.get("Description") or [""])[0].strip()
    job_levels = _split_csv((fields.get("Job levels") or [""])[0])
    languages = _split_csv((fields.get("Languages") or [""])[0])

    duration_text = ""
    duration_minutes: int | None = None
    untimed = False
    for line in fields.get("Assessment length", []):
        if "Completion Time" in line or "minute" in line.lower() or "untimed" in line.lower():
            duration_text = line
            untimed = "untimed" in line.lower()
            m = re.search(r"(\d+)", line)
            if m:
                duration_minutes = int(m.group(1))
            break

    return {
        "description": description,
        "job_levels": job_levels,
        "languages": languages,
        "duration_text": duration_text,
        "duration_minutes": duration_minutes,
        "untimed": untimed,
    }


def _split_csv(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def fetch(client: httpx.Client, url: str, *, retries: int = 3) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = client.get(url, timeout=30.0)
            r.raise_for_status()
            return r.text
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def crawl_listing(client: httpx.Client) -> list[dict]:
    """Walk start=0,12,... until an Individual Test Solutions table comes back empty."""
    items: list[dict] = []
    seen_urls: set[str] = set()
    start = 0
    empty_streak = 0
    while True:
        url = f"{LISTING_URL}?start={start}&type=1"
        html = fetch(client, url)
        rows = parse_listing_rows(html)
        new = [r for r in rows if r["url"] not in seen_urls]
        if not new:
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0
            for r in new:
                seen_urls.add(r["url"])
                items.append(r)
        print(f"  start={start}: page returned {len(rows)}, new {len(new)}, total {len(items)}")
        start += PAGE_SIZE
        # Safety guard
        if start > 1000:
            print("  hit safety guard at start>1000, stopping")
            break
    return items


def enrich(client: httpx.Client, listing_items: list[dict]) -> list[Assessment]:
    enriched: list[Assessment] = []
    for i, item in enumerate(listing_items, 1):
        try:
            html = fetch(client, item["url"])
            detail = parse_detail(html)
        except Exception as exc:
            print(f"  [{i}/{len(listing_items)}] FAIL {item['name']}: {exc}")
            detail = {
                "description": "",
                "job_levels": [],
                "languages": [],
                "duration_text": "",
                "duration_minutes": None,
                "untimed": False,
            }
        a = Assessment(
            name=item["name"],
            url=item["url"],
            slug=item["slug"],
            test_types=item["test_types"],
            test_type_labels=[
                TEST_TYPE_LABELS.get(c, c) for c in item["test_types"]
            ],
            remote_testing=item["remote_testing"],
            adaptive_irt=item["adaptive_irt"],
            **detail,
        )
        enriched.append(a)
        if i % 10 == 0 or i == len(listing_items):
            print(f"  detail [{i}/{len(listing_items)}] {item['name'][:60]}")
    return enriched


def main() -> int:
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        print("Phase 1: crawling listing pages")
        listing = crawl_listing(client)
        print(f"  -> {len(listing)} unique Individual Test Solutions found")
        if not listing:
            print("ERROR: no items found", file=sys.stderr)
            return 1

        print("\nPhase 2: fetching detail pages")
        records = enrich(client, listing)

    payload = [asdict(r) for r in records]
    OUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(payload)} records to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
