import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import certifi

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def is_valid_chaitanya_charan_track(url: str) -> bool:
    """
    Accepts only:
    https://soundcloud.com/chaitanya-charan/<track-slug>
    """
    try:
        parsed = urlparse(url)
        if parsed.netloc != "soundcloud.com":
            return False

        parts = parsed.path.strip("/").split("/")
        return len(parts) == 2 and parts[0] == "chaitanya-charan"
    except Exception:
        return False


def extract_page_content(page_url: str, title: str) -> dict:
    response = requests.get(page_url, headers=HEADERS, timeout=20, verify=False)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    container = soup.find("div", class_="entry-content single-content")
    if not container:
        raise ValueError("entry-content single-content div not found")

    # Remove Similar Posts block
    for unwanted in container.select(".crp_related"):
        unwanted.decompose()

    transcripts = []
    soundcloud_links = set()
    youtube_links = set()

    # ----------------------------
    # Extract iframe links (YouTube only)
    # ----------------------------
    for iframe in container.find_all("iframe"):
        src = iframe.get("src")
        if not src:
            continue

        full_url = urljoin(page_url, src)

        if is_youtube(full_url):
            youtube_links.add(full_url)

    # ----------------------------
    # Extract anchor links
    # ----------------------------
    for a in container.find_all("a", href=True):
        href = urljoin(page_url, a["href"])

        if is_valid_chaitanya_charan_track(href):
            soundcloud_links.add(href)

        elif is_youtube(href):
            youtube_links.add(href)

    # ----------------------------
    # Extract transcription text
    # ----------------------------
    transcription_start = None

    for tag in container.find_all(["b", "strong"]):
        if "transcription" in tag.get_text(strip=True).lower():
            transcription_start = tag
            break

    if transcription_start:
        # Case 1: Explicit transcription section
        for elem in transcription_start.find_all_next("p"):
            text = elem.get_text(strip=True)
            if not text:
                continue
            if "end of transcription" in text.lower():
                break
            transcripts.append(text)
    else:
        # Case 2: No transcription marker → extract all <p>
        for p in container.find_all("p"):
            text = p.get_text(strip=True)
            if text:
                transcripts.append(text)

    return {
        "url": page_url,
        "title": title,
        "transcript": "\n\n".join(transcripts),
        "soundcloud_links": sorted(soundcloud_links),
        "youtube_links": sorted(youtube_links),
    }


if __name__ == "__main__":
    url = "https://www.thespiritualscientist.com/how-can-humility-go-along-with-self-respect/"

    title = "Humily and self respect"

    data = extract_page_content(url, title)

    print(data)

    print("\n--- TRANSCRIPT ---\n")
    print(data["transcript"][:1500], "...\n")

    print("\n--- SOUNDCLOUD LINKS ---")
    for link in data["soundcloud_links"]:
        print(link)

    print("\n--- YOUTUBE LINKS ---")
    for link in data["youtube_links"]:
        print(link)
