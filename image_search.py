import math
import requests
import pandas as pd
import numpy as np
from urllib.parse import urljoin
from io import BytesIO
from PIL import Image
import re
from transformers import BlipProcessor, BlipForConditionalGeneration

REDDIT_BASE = "https://www.reddit.com"
SUBREDDIT = "memes"
LIMIT = 50
TIMEFRAME = "day"   
OUTPUT_XLSX = "memes_top50.xlsx"

#Header for reddit API request
HEADERS = {
    "User-Agent": "script:memes-top50:v2.1 (by u/rami)"
}

REDDIT_EPOCH = 1134028003
VIRAL_PERCENTILE = 80

#Stopwords for keyword extraction from captions
STOPWORDS = {
    "a","an","the","and","or","but","if","while","with","without","on","in","at","to","from","by","for","of",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","as","into","over",
    "about","above","below","up","down","off","out","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","can","will","just","don","should","now","you","your","yours",
    "me","my","mine","we","our","ours","they","them","their","theirs","he","him","his","she","her","hers","i"
}

_BLIP_PROCESSOR = None
_BLIP_MODEL = None
#Load BLIP captioning model
def _load_blip():
    global _BLIP_PROCESSOR, _BLIP_MODEL
    if _BLIP_PROCESSOR is None or _BLIP_MODEL is None:
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _BLIP_PROCESSOR, _BLIP_MODEL

#Get first image URL from a Reddit gallery post
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

#Guess a usable image URL from a Reddit post
def reddit_image_url_post(post):
    if post.get("is_video"):
        return ""
    url = post.get("url_overridden_by_dest") or post.get("url")
    domain = post.get("domain", "")
    post_hint = post.get("post_hint", "")
    video_domains = ["v.redd.it", "youtube.com", "youtu.be", "tiktok.com"]
    if any(d in domain for d in video_domains):
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
        if src.get("url"):
            return src["url"].replace("&amp;", "&")
    return ""

#Fetch top posts from subreddit using Reddit JSON API
def fetch_top_posts():
    url = f"{REDDIT_BASE}/r/{SUBREDDIT}/top.json"
    params = {"limit": LIMIT, "t": TIMEFRAME}
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return [c.get("data", {}) for c in data.get("data", {}).get("children", [])]

#Compute hot score for a post
def hot_score_from_post(post):
    score = post.get("score", post.get("ups", 0))
    created = post.get("created_utc", 0)
    s = 1 if score > 0 else (-1 if score < 0 else 0)
    order = math.log10(max(abs(score), 1))
    seconds = created - REDDIT_EPOCH
    return order + s * (seconds / 45000.0)

#Generate a BLIP caption for an PIL (pillow) image
def blip_caption(image):
    try:
        processor, model = _load_blip()
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""
#Download image from URL and return as PIL.Image
def fetch_image(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
#Convert a caption string into short keyword string
def caption_to_keywords(caption: str) -> str:
    if not caption:
        return ""
    words = [w.lower() for w in _WORD_RE.findall(caption)]
    kept = []
    for w in words:
        if w in STOPWORDS or len(w) <= 2:
            continue
        if w not in kept:
            kept.append(w)
    kept_sorted = sorted(kept, key=lambda w: (-len(w), w))
    top5 = kept_sorted[:5]  
    return ", ".join(top5)

def extract_image_keywords(image_url: str) -> str:
    img = fetch_image(image_url)
    if img is None:
        return ""
    caption = blip_caption(img)
    return caption_to_keywords(caption)

#Build table rows in excel sheet for image data
def build_rows(posts):
    rows = []
    for p in posts:
        image_link = reddit_image_url_post(p)
        if not image_link:
            continue
        hot = hot_score_from_post(p)
        keywords = extract_image_keywords(image_link)
        rows.append({
            #Image details
            "post link": urljoin(REDDIT_BASE, p.get("permalink", "")),
            "image link": image_link,
            "# of upvotes": p.get("ups", 0),
            "# of comments": p.get("num_comments", 0),
            "hot": hot,
            "image keywords": keywords,
        })
    return rows

def main():
    posts = fetch_top_posts()
    rows = build_rows(posts)
    if not rows:
        print("No image posts found.")
        return
    df = pd.DataFrame(rows)
    cutoff = np.percentile(df["hot"], VIRAL_PERCENTILE)
    df["viral"] = np.where(df["hot"] >= cutoff, "YES", "NO")
    df_out = df[[
        #output columns for excel sheet
        "post link", "image link", "# of upvotes", "# of comments",
        "image keywords", "viral"
    ]]
    #Writes to excel file
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="r_memes_top50")
    print(f"Wrote {len(df_out)} image posts to {OUTPUT_XLSX}")
    print(f"Viral cutoff (>= {VIRAL_PERCENTILE}th percentile hot) = {cutoff:.6f}")

if __name__ == "__main__":
    main()
