"""One-shot probe to dump real HTML from SHL so we can pick selectors."""
import httpx
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "probe"
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "listing_p1.html": "https://www.shl.com/products/product-catalog/?start=0&type=1",
    "listing_p2.html": "https://www.shl.com/products/product-catalog/?start=12&type=1",
    "detail_opq32r.html": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    "detail_verify_g.html": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def main() -> None:
    with httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
        for name, url in URLS.items():
            print(f"GET {url}")
            r = client.get(url)
            print(f"  -> {r.status_code} {len(r.text)} bytes")
            (OUT / name).write_text(r.text, encoding="utf-8")
    print(f"\nSaved probes to {OUT}")


if __name__ == "__main__":
    main()
