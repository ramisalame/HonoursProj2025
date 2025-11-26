import re
from urllib.parse import urljoin

import requests

from config import (
    REDDIT_BASE,
    SUBREDDIT,
    LIMIT,
    TIMEFRAME,
    HEADERS,
    MAX_IMAGES,
    SMALL_SUBSCRIBERS_THRESHOLD,
    hot_score_from_post,
)
from image_semantics import (
    extract_image_keywords_with_scores,
    classify_subreddit_theme,
)


# -------------------------------------------------------------------
# Reddit 
# -------------------------------------------------------------------
def first_gallery_image(post):
    media = post.get("media_metadata", {})
    order = post.get("gallery_data", {}).get("items", [])
    if not media or not order:
        return None
    for item in order:
        meta = media.get(item.get("media_id", ""), {})
        src = meta.get("s", {})
        url = src.get("u") or src.get("gif") or src.get("mp4")
        if url:
            return url.replace("&amp;", "&")
    return None


def guess_image_url(post):
    if post.get("is_video"):
        return ""
    url = post.get("url_overridden_by_dest") or post.get("url")
    domain = post.get("domain", "")
    post_hint = post.get("post_hint", "")

    if any(d in domain for d in ["v.redd.it", "youtube.com", "youtu.be", "tiktok.com"]):
        return ""

    if post.get("is_gallery"):
        g = first_gallery_image(post)
        if g:
            return g

    if post_hint == "image" and url:
        return url

    if url and any(domain.endswith(d) for d in ["i.redd.it", "i.imgur.com", "preview.redd.it"]):
        return url

    preview = post.get("preview", {})
    images = preview.get("images", [])
    if images:
        src = images[0].get("source", {})
        u = src.get("url")
        if u:
            return u.replace("&amp;", "&")

    return ""


def fetch_top_posts():
    r = requests.get(
        f"{REDDIT_BASE}/r/{SUBREDDIT}/top.json",
        headers=HEADERS,
        params={"limit": LIMIT, "t": TIMEFRAME},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    return [c.get("data", {}) for c in data.get("data", {}).get("children", [])]


def fetch_reddit_post_by_url(url: str, timeout: int = 10):
    try:
        base = url.split("?", 1)[0].rstrip("/")
        api_url = base + ".json"
        r = requests.get(api_url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        js = r.json()

        if isinstance(js, list) and js:
            listing = js[0].get("data", {}).get("children", [])
            if listing:
                return listing[0].get("data", {}) or None

        if isinstance(js, dict):
            data = js.get("data", {})
            if isinstance(data, dict) and "children" in data:
                children = data.get("children", [])
                if children:
                    return children[0].get("data", {}) or None
            return data or None
    except Exception as e:
        print(f"[warn] Failed to fetch reddit post for URL {url}: {e}")
    return None


def get_keywords_for_reddit_url(url: str, want_n: int = 6):

    if "reddit.com" not in (url or ""):
        return "", ""

    post = fetch_reddit_post_by_url(url)
    if not post:
        return "", ""

    img_link = guess_image_url(post)
    if not img_link:
        return "", ""

    title = post.get("title", "") or ""
    subreddit = post.get("subreddit", "") or ""
    selftext = (post.get("selftext", "") or "")[:300]

    context = f"subreddit r/{subreddit}. title: {title}. text: {selftext}"

    tags, _scores = extract_image_keywords_with_scores(
        img_link,
        want_n=want_n,
        context_text=context,
    )

    return img_link, ",".join(tags)


def fetch_reddit_sub_about(subreddit: str, timeout=10) -> dict:
    info = {
        "display_name_prefixed": f"r/{subreddit}",
        "title": "",
        "public_description": "",
        "subscribers": None,
    }
    try:
        url = f"{REDDIT_BASE}/r/{subreddit}/about.json"
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json().get("data", {})
        else:
            data = {}

        info["display_name_prefixed"] = data.get(
            "display_name_prefixed", info["display_name_prefixed"]
        )
        info["title"] = data.get("title", "")
        info["public_description"] = (
            data.get("public_description", "") or data.get("community_icon_alt", "")
        )
        subs = data.get("subscribers")
        if isinstance(subs, int) and subs >= 0:
            info["subscribers"] = subs
    except Exception:
        pass
    return info


def extract_subreddit_from_url(url: str) -> str | None:
    from urllib.parse import urlparse

    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        if "reddit.com" not in host:
            return None
        parts = [s for s in p.path.split("/") if s]
        for i, seg in enumerate(parts):
            if seg == "r" and i + 1 < len(parts):
                sub = parts[i + 1]
                sub = re.sub(r"[^A-Za-z0-9_+-]", "", sub)
                return sub if sub else None
    except Exception:
        pass
    return None


def _bucket_from_subscribers(subscribers: int | None) -> str:
    if subscribers is None:
        return ""
    return "smaller" if subscribers < SMALL_SUBSCRIBERS_THRESHOLD else "larger"


def enrich_with_reddit_meta(result_row: dict) -> dict:
    url = result_row.get("result_url", "")
    sub = extract_subreddit_from_url(url)
    if not sub:
        result_row.setdefault("subreddit", "")
        result_row.setdefault("reddit community about", "")
        result_row.setdefault("subreddit subscribers", None)
        result_row.setdefault("community size bucket", "")
        return result_row

    about = fetch_reddit_sub_about(sub)
    desc = about.get("public_description", "") or about.get("title", "")
    category = classify_subreddit_theme(sub, about.get("title", ""), desc)
    subs = about.get("subscribers", None)
    size_bucket = _bucket_from_subscribers(subs)

    result_row["subreddit"] = about.get("display_name_prefixed", f"r/{sub}")
    result_row["reddit community about"] = category
    result_row["subreddit subscribers"] = subs
    result_row["community size bucket"] = size_bucket
    return result_row


# -------------------------------------------------------------------
# Building rows for posts with meme tags
# -------------------------------------------------------------------
def build_rows(posts):
    rows = []
    for p in posts:
        image_link = guess_image_url(p)
        if not image_link:
            continue

        hot = hot_score_from_post(p)

        title = p.get("title", "") or ""
        subreddit = p.get("subreddit", "") or ""
        selftext = (p.get("selftext", "") or "")[:300]

        context = f"subreddit r/{subreddit}. title: {title}. text: {selftext}"

        tags, tag_scores = extract_image_keywords_with_scores(
            image_link,
            want_n=6,
            context_text=context
        )

        if tag_scores:
            print("\n[Keyword scores for]", title)
            for t, s in tag_scores:
                print(f"  {t:35s} {s:5.2f}%")

        rows.append({
            "post": p,
            "post link": urljoin(REDDIT_BASE, p.get("permalink", "")),
            "image link": image_link,
            "# of upvotes": p.get("ups", 0),
            "# of comments": p.get("num_comments", 0),
            "hot": hot,
            "image keywords": ",".join(tags),
            "image keyword scores": ",".join([f"{t}:{s:.2f}%" for t, s in tag_scores]),
        })

        if len(rows) >= MAX_IMAGES:
            break

    return rows
