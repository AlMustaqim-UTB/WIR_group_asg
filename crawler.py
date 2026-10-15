from collections import deque
from time import monotonic, sleep
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


base_url = "https://bulbapedia.bulbagarden.net/wiki/Main_Page"
user_agent = "WIRGroupCrawler/1.0"

base_parts = urlsplit(base_url)
allowed_netloc = base_parts.netloc

session = requests.Session()
session.headers.update({"User-Agent": user_agent})

robots_url = urlunsplit(
    (base_parts.scheme, base_parts.netloc, "/robots.txt", "", "")
)

robots_response = session.get(robots_url, timeout=30)
robots_response.raise_for_status()

robots = RobotFileParser()
robots.set_url(robots_url)
robots.parse(robots_response.text.splitlines())

crawl_delay = robots.crawl_delay(user_agent)
if crawl_delay is None:
    crawl_delay = robots.crawl_delay("*")
if crawl_delay is None:
    crawl_delay = 0

visited_urls = set()
urls_to_visit = deque([base_url])
last_request_time = 0.0


def normalize_url(url):
    """Remove fragments and normalize the URL before queueing it."""
    url, _fragment = urldefrag(url)
    parts = urlsplit(url)

    if parts.scheme not in {"http", "https"}:
        return None

    if parts.netloc != allowed_netloc:
        return None

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path or "/",
            parts.query,
            "",
        )
    )


def wait_for_crawl_delay():
    global last_request_time

    elapsed = monotonic() - last_request_time
    if elapsed < crawl_delay:
        sleep(crawl_delay - elapsed)

    last_request_time = monotonic()


def crawl_page(url):
    if not robots.can_fetch(user_agent, url):
        print(f"Skipped by robots.txt: {url}")
        return []

    wait_for_crawl_delay()

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        print(f"Error crawling {url}: {error}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    links = []

    for link in soup.find_all("a", href=True):
        next_url = normalize_url(urljoin(url, link["href"]))

        if (
            next_url is not None
            and next_url not in visited_urls
            and next_url not in urls_to_visit
            and robots.can_fetch(user_agent, next_url)
        ):
            links.append(next_url)

    return links


while urls_to_visit:
    current_url = urls_to_visit.popleft()

    if current_url in visited_urls:
        continue

    visited_urls.add(current_url)
    print(f"Crawling: {current_url}")

    new_links = crawl_page(current_url)
    urls_to_visit.extend(new_links)

print("Crawling finished.")