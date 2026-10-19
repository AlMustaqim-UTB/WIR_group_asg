import os
import json
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from collections import deque
import requests
from bs4 import BeautifulSoup

# testing
def setup_robot_parser(base_url: str, user_agent: str) -> RobotFileParser:
    """Fetch and parse the robots.txt file for politeness checking."""
    robots_url = (urljoin(base_url, "/robots.txt"))
    headers = {"User-Agent": user_agent}
    robot_parser = RobotFileParser()

    try:
        response = requests.get(robots_url, headers=headers, timeout=10)
        if response.status_code == 200:
            robot_parser.parse(response.text.splitlines())
            print(f"[robots.txt] Successfully loaded from {robots_url}")
        else:
            print(f"[robots.txt] Could not retrieve (Status: {response.status_code}). Assuming allowed.")
    except Exception as e:
        print(f"[robots.txt] Error fetching robots.txt: {e}. Assuming allowed.")

    return robot_parser


def crawl_domain(seed_url: str, target_count: int = 50, output_dir: str = "crawled_documents"):
    """
    Crawls pages starting from seed_url within the same domain until
    at least `target_count` pages are collected or the queue is exhausted.
    """
    user_agent = "StandardPythonCrawler/1.0"
    headers = {"User-Agent": user_agent}

    # Derive base domain and scheme
    parsed_seed = urlparse(seed_url)
    allowed_netloc = parsed_seed.netloc
    base_url = f"{parsed_seed.scheme}://{allowed_netloc}"

    # Setup robots.txt
    robot_parser = setup_robot_parser(base_url, user_agent)

    # Prepare document output folder
    os.makedirs(output_dir, exist_ok=True)

    # Crawler state
    queue = deque([seed_url])
    visited = set()
    records = []
    doc_id_counter = 1

    # Ignored extensions for web documents
    ignored_extensions = (
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
        ".pdf", ".zip", ".tar", ".gz", ".mp3", ".mp4", ".css", ".js"
    )

    print(f"\nStarting crawler on domain: {allowed_netloc}")

    while queue and doc_id_counter <= target_count:
        current_url = queue.popleft()

        # Remove URL anchors/fragments (e.g. #cite_note-1)
        current_url, _ = urldefrag(current_url)
        if current_url in visited:
            continue
        visited.add(current_url)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check robots.txt permissions
        if not robot_parser.can_fetch(user_agent, current_url):
            print(f"[-] Blocked by robots.txt: {current_url}")
            records.append({
                "Document_ID": "N/A",
                "Source_URL": current_url,
                "Crawl_Date": timestamp,
            })
            continue

        status_code = None
        collected = "No"

        try:
            # Respectful request interval
            time.sleep(0.5)

            response = requests.get(current_url, headers=headers, timeout=10)
            status_code = response.status_code

            # Only process successful HTML responses
            content_type = response.headers.get("Content-Type", "")
            if status_code == 200 and "text/html" in content_type:
                # 1. Save document content locally
                filename = os.path.join(output_dir, f"doc_{doc_id_counter:03d}.html")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(response.text)

                collected = "Yes"
                print(f"[+] [{doc_id_counter}/{target_count}] Collected: {current_url}")

                # 2. Extract internal links to expand crawler queue
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"].strip()
                    # Resolve relative URLs
                    full_url = urljoin(current_url, href)
                    full_url, _ = urldefrag(full_url)
                    parsed_link = urlparse(full_url)

                    # Domain constraint and filter non-document links
                    if (
                        parsed_link.netloc == allowed_netloc
                        and full_url not in visited
                        and not any(parsed_link.path.lower().endswith(ext) for ext in ignored_extensions)
                        and not full_url.endswith("/Special:Random")  # Avoid non-deterministic redirect loops
                    ):
                        queue.append(full_url)

        except requests.RequestException as e:
            print(f"[!] Request error fetching {current_url}: {e}")
            status_code = "RequestError"

        # Record metadata for this URL
        records.append({
            "Document_ID": f"DOC_{doc_id_counter:03d}" if collected == "Yes" else "N/A",
            "Source_URL": current_url,
            "Crawled_Date": timestamp,
        })

        if collected == "Yes":
            doc_id_counter += 1

    # Save metadata log into JSON format
    json_output_path = "crawl_results.json"
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(records, json_file, indent=4, ensure_ascii=False)

    print("-" * 100)
    print(f"Crawl completed!")
    print(f"Total documents successfully collected: {doc_id_counter - 1}")
    print(f"HTML documents saved in: '{output_dir}/'")
    print(f"Crawl summary saved in: '{json_output_path}'")


if __name__ == "__main__":
    seed_url = "https://minecraft.wiki/w/Block"
    crawl_domain(seed_url=seed_url, target_count=100)