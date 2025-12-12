import random
import re
from urllib.parse import urlsplit, parse_qs, quote, urlparse

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from config import (
    GOOGLE_USER_AGENT,
    SELENIUM_HEADLESS,
    SELENIUM_PAGELOAD_TIMEOUT,
    SELENIUM_WAIT_S,
    MAX_RESULTS_PER_IMAGE,
    SMALL_SUBSCRIBERS_THRESHOLD,
    sleep,
)
from image_semantics import TAG_CATEGORY
from reddit_utils import (
    get_keywords_for_reddit_url,
    enrich_with_reddit_meta,
)


# -------------------------------------------------------------------
# Selenium / Google Lens reverse image search
# -------------------------------------------------------------------
def make_driver():
    opts = Options()
    #headless chrome page
    if SELENIUM_HEADLESS:
        opts.add_argument("--headless=new")
    else:
        opts.add_experimental_option("detach", True)

    from config import CHROME_USER_DATA_DIR, CHROME_PROFILE_DIR
    
    #local Chrome profile
    if CHROME_USER_DATA_DIR and CHROME_USER_DATA_DIR.strip():
        opts.add_argument(f"--user-data-dir={CHROME_USER_DATA_DIR}")
        if CHROME_PROFILE_DIR:
            opts.add_argument(f"--profile-directory={CHROME_PROFILE_DIR}")

    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,2000")
    opts.add_argument(f"--user-agent={GOOGLE_USER_AGENT}")

    service = ChromeDriverManager().install()
    driver = webdriver.Chrome(service=webdriver.ChromeService(service), options=opts)
    #page load timeout if page stops loading 
    driver.set_page_load_timeout(SELENIUM_PAGELOAD_TIMEOUT)
    return driver


def lens_url_for_image(image_url: str) -> str:
    return f"https://lens.google.com/uploadbyurl?url={quote(image_url, safe='')}&hl=en"


def get_visible_text(driver) -> str:
    try:
        return (driver.execute_script(
            "return document.body && document.body.innerText ? document.body.innerText : '';"
        ) or "").lower()
    except Exception:
        return (driver.page_source or "").lower()


#Detect whether the current Google Lens page is showing a CAPTCHA
def is_captcha_screen(driver) -> bool:
    txt = get_visible_text(driver)
    needles = [
        "our systems have detected unusual traffic from your computer network",
        "to continue, please type the characters",
        "verify that you are human",
        "sorry, we can't process your request right now",
    ]
    return any(n in txt for n in needles)



#Wait for result links to appear on the page or timeout
def has_any_result_links(driver) -> bool:
    try:
        anchors = driver.find_elements(By.XPATH, "//a[@href]")
        return len(anchors) >= 5
    except Exception:
        return False


def wait_for_results_or_timeout(driver, timeout_s=12):
    try:
        WebDriverWait(driver, timeout_s).until(
            EC.presence_of_element_located((By.XPATH, "//a[@href]"))
        )
    except TimeoutException:
        pass

#Wait for captcha to clear 
def wait_for_captcha_clear(driver, max_wait_s=300):
    print("CAPTCHA detected.")
    waited = 0
    interval = 3
    while waited < max_wait_s:
        sleep(interval)
        waited += interval
        try:
            if not is_captcha_screen(driver):
                print("CAPTCHA cleared")
                return True
        except Exception:
            pass
    print("Skip image.")
    return False


#Click exact match filter tab in google reverse search if available
def click_exact_tab(driver):
    candidates = [
        (By.XPATH, "//span[text()='Exact']"),
        (By.XPATH, "//span[contains(., 'Exact')]"),
        (By.XPATH, "//button[.//span[contains(., 'Exact')]]"),
        (By.CSS_SELECTOR, "[aria-label*='Exact']"),
        (By.XPATH, "//*[contains(translate(., 'EXACT', 'exact'), 'exact')]"),
    ]
    for by, sel in candidates:
        try:
            el = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((by, sel)))
            driver.execute_script("arguments[0].click();", el)
            sleep(1.0)
            return True
        except Exception:
            continue
    return False

#Extract target url from a given result link 
def _extract_target_from_google_href(href: str) -> str | None:
    try:
        u = urlsplit(href)
        if "google." not in (u.netloc or ""):
            return href
        qs = parse_qs(u.query)
        if "q" in qs and qs["q"]:
            return qs["q"][0]
        if "imgrefurl" in qs and qs["imgrefurl"]:
            return qs["imgrefurl"][0]
        if u.path.endswith("/url") or u.path.endswith("/imgres"):
            return qs.get("q", qs.get("imgrefurl", [None]))[0]
    except Exception:
        pass
    return None


#Extract contents from google lens results page
def extract_result_links_from_lens(driver, max_results=MAX_RESULTS_PER_IMAGE):
    urls = []
    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.XPATH, "//a[@href]"))
        )
    except TimeoutException:
        pass
    
    #Scrolls to reach end of page
    for _ in range(6):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(0.8 + random.uniform(0.0, 0.6))

    wait_for_results_or_timeout(driver, timeout_s=6)
    anchors = driver.find_elements(By.XPATH, "//a[@href]")

    #Extracts only reddit links
    for a in anchors:
        href = a.get_attribute("href") or ""
        if not href:
            continue
        target = _extract_target_from_google_href(href)
        if not target:
            continue
        if "reddit.com" not in target:
            continue
        urls.append(target)

    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_results:
            break
    return out


DOMAIN_LABELS = {
    "reddit.com": "Reddit — discussion communities/subreddits",
    "old.reddit.com": "Reddit — discussion communities/subreddits",
    "i.redd.it": "Reddit image host",
}

#Classify domain into a type
def classify_domain(url: str, fallback_title: str = "") -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host in DOMAIN_LABELS:
            return DOMAIN_LABELS[host]
        for k, v in DOMAIN_LABELS.items():
            if host.endswith(k):
                return v
    except Exception:
        pass

    #Different website types
    title = (fallback_title or "").lower()
    if any(k in title for k in ["forum", "community", "discussion"]):
        return "Online forum/community"
    if any(k in title for k in ["blog", "medium"]):
        return "Blog/publishing platform"
    if any(k in title for k in ["news", "press"]):
        return "News/media site"
    if "wiki" in title:
        return "Wiki / knowledge base"
    if any(k in title for k in ["store", "shop", "cart"]):
        return "E-commerce site"
    return "Website"


#Fetches the title of the url
def fetch_title_of_url(url: str, timeout=10) -> str:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": GOOGLE_USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )
        r.raise_for_status()
        html = r.text[:200000]
        t = BeautifulSoup(html, "lxml").find("title")
        return re.sub(r"\s+", " ", t.text).strip() if (t and t.text) else ""

    #Fallback 
    except Exception:
        return ""


def extract_subreddit_from_url(url: str) -> str | None:
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


#Runs google lens "exact search" for given original image url 
def google_reverse_image_exact(image_url: str, driver) -> list[dict]:

    #google lens "upload by url"
    url = lens_url_for_image(image_url)

    #random sleep
    sleep(2.0 + random.uniform(0.0, 1.0))

    
    from selenium.common.exceptions import TimeoutException as SeleniumTimeout

    try:
        #open lens url 
        driver.get(url)
    except SeleniumTimeout:
        sleep(2.0 + random.uniform(0.0, 1.0))
        try:
            driver.get(url)
        except Exception:
            return []
    #Wait to let lens load the page with the image
    sleep(SELENIUM_WAIT_S + random.uniform(0.2, 0.8))
    wait_for_results_or_timeout(driver, timeout_s=12)

    #Wait for user to solve captcha 
    if is_captcha_screen(driver) and not has_any_result_links(driver):
        if not wait_for_captcha_clear(driver, max_wait_s=300):
            return []
        sleep(1.0)

    #Click on exact tab 
    try:
        click_exact_tab(driver)
        sleep(SELENIUM_WAIT_S)
    except Exception:
        pass

    #Scroll to bottom of page
    for _ in range(6):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(0.8 + random.uniform(0.0, 0.6))

    wait_for_results_or_timeout(driver, timeout_s=6)
    links = extract_result_links_from_lens(driver, max_results=MAX_RESULTS_PER_IMAGE)

    out = []
    for link in links:
        #fetch title of result page
        title = fetch_title_of_url(link)

        #classify domain
        label = classify_domain(link, fallback_title=title)

        match_image_link = ""
        match_image_keywords = ""

        if "reddit.com" in (link or ""):
            match_image_link, match_image_keywords = get_keywords_for_reddit_url(link, want_n=6)

        row = {
            "result_url": link,
            "community_about": label if label else (title if title else "Website"),
            "page_title": title,
            "match_image_link": match_image_link,
            "match_image_keywords": match_image_keywords,
        }

        #subreddit metadata subname, subscribers, subs count)
        row = enrich_with_reddit_meta(row)
        out.append(row)
    return out


# -------------------------------------------------------------------
# Excel helpers + analysis
# -------------------------------------------------------------------

#Specify if image is in smaller community or larger community or in between
def _summarize_size_tendency(rows_for_image: list[dict]) -> str:
    if not rows_for_image:
        return "unknown"
    buckets = [r.get("community size bucket", "") for r in rows_for_image if r.get("community size bucket", "")]
    if not buckets:
        return "unknown"
    small = sum(1 for b in buckets if b == "smaller")
    large = sum(1 for b in buckets if b == "larger")
    if small > large:
        return "mostly smaller"
    if large > small:
        return "mostly larger"
    return "mixed"


def summarize_size_tendencies(reverse_rows: list[dict]) -> dict:
    
    #Build a mapping of'mostly smaller' / 'mostly larger' / 'mixed' / 'unknown'

    tendency_map = {}
    by_image = {}
    for row in reverse_rows:
        by_image.setdefault(row.get("original image link", ""), []).append(row)
    for img, rows_for_img in by_image.items():
        tendency_map[img] = _summarize_size_tendency(rows_for_img)
    return tendency_map

#Write/overwrite the top sheet (r_memes_top5)
def write_top_sheet(df_out: pd.DataFrame, path: str):
    mode = "a" if pd.io.common.file_exists(path) else "w"
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
        df_out.to_excel(writer, index=False, sheet_name="r_memes_top5")


#Write to the excel file with search results 
def append_reverse_sheet(reverse_rows: list[dict], path: str):
    df_rev = pd.DataFrame(reverse_rows).rename(columns={
        "original_image_link": "original image link",
        "result_url": "result url",
        "community_about": "community about",
        "page_title": "page title",
        "subreddit": "subreddit",
        "reddit community about": "reddit community about",
        "subreddit subscribers": "subreddit subscribers",
        "community size bucket": "community size bucket",
        "match_image_link": "result image link",
        "match_image_keywords": "result image keywords",
    })

    for col in [
        "subreddit",
        "reddit community about",
        "subreddit subscribers",
        "community size bucket",
        "result image link",
        "result image keywords",
    ]:
        if col not in df_rev.columns:
            if col == "subreddit subscribers":
                df_rev[col] = None
            else:
                df_rev[col] = ""

    cols = [
        "original image link",
        "result url",
        "page title",
        "community about",
        "subreddit",
        "subreddit subscribers",
        "community size bucket",
        "reddit community about",
        "result image link",
        "result image keywords",
    ]
    df_rev = df_rev[[c for c in cols if c in df_rev.columns]]

    mode = "a" if pd.io.common.file_exists(path) else "w"
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
        df_rev.to_excel(writer, index=False, sheet_name="reverse_search")


#Analyze meme stats based off data
def analyze_meme_popularity(df_out: pd.DataFrame):
    if "image keywords" not in df_out.columns:
        print("No 'image keywords' column found")
        return

    records = []
    for _, row in df_out.iterrows():
        upvotes = row.get("# of upvotes", 0)
        viral = row.get("viral", "NO")
        kw_str = row.get("image keywords", "") or ""
        tags = [t.strip().lower() for t in kw_str.split(",") if t.strip()]
        for tag in tags:
            category = TAG_CATEGORY.get(tag, "other / unknown")
            records.append({
                "tag": tag,
                "category": category,
                "upvotes": upvotes,
                "viral": 1 if str(viral).upper() == "YES" else 0,
            })

    if not records:
        print("No tags found in image keywords")
        return

    df_long = pd.DataFrame(records)
    #Stats at category level
    cat_stats = (
        df_long
        .groupby("category", as_index=False)
        .agg(
            mean_upvotes=("upvotes", "mean"),
            median_upvotes=("upvotes", "median"),
            count=("tag", "size"),
            viral_rate=("viral", "mean"),
        )
        .sort_values("mean_upvotes", ascending=False)
    )

    
    print("\n=== Meme Category Popularity ===")
    print(cat_stats.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    print("================================\n")

    if cat_stats.empty:
        return
    
    #Identify the best-performing category
    top_cat = cat_stats.iloc[0]["category"]
    print(f"Most popular category by average upvotes: {top_cat}")

    df_top_cat = df_long[df_long["category"] == top_cat]
    tag_stats = (
        df_top_cat
        .groupby("tag", as_index=False)
        .agg(
            mean_upvotes=("upvotes", "mean"),
            count=("tag", "size"),
            viral_rate=("viral", "mean"),
        )
        .sort_values("mean_upvotes", ascending=False)
    )

    print(f"\n=== Top tags in category '{top_cat}' ===")
    print(tag_stats.head(10).to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    print("========================================\n")

    #average upvotes by category bar chart 
    plt.figure(figsize=(10, 6))
    plt.bar(cat_stats["category"], cat_stats["mean_upvotes"])
    plt.xlabel("Meme Category")
    plt.ylabel("Average upvotes")
    plt.title("Average Upvotes by Meme Category")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def analyze_small_community_keywords(reverse_rows: list[dict], top_n: int = 15):
   
    #Reverse search results and find which meme tags (match_image_keywords) show up in smaller communities
    if not reverse_rows:
        print("No reverse search rows")
        return

    records = []
    for r in reverse_rows:
        subs = r.get("subreddit subscribers")
        kw_str = r.get("match_image_keywords", "") or ""

        if subs is None:
            continue
        try:
            subs_int = int(subs)
        except (TypeError, ValueError):
            continue

        if subs_int >= SMALL_SUBSCRIBERS_THRESHOLD:
            continue

        if not kw_str:
            continue

        tags = [t.strip().lower() for t in kw_str.split(",") if t.strip()]
        for tag in tags:
            category = TAG_CATEGORY.get(tag, "other / unknown")
            records.append({"tag": tag, "category": category})

    if not records:
        print("No tags found in smaller communities")
        return

    df_small = pd.DataFrame(records)
    # Count of how often each tag appears in small communities
    tag_counts = (
        df_small
        .groupby("tag", as_index=False)
        .agg(count=("tag", "size"))
        .sort_values("count", ascending=False)
    )

    print("\n=== Keywords in smaller communities "
          f"(< {SMALL_SUBSCRIBERS_THRESHOLD:,} subscribers) ===")
    print(tag_counts.head(top_n).to_string(index=False))
    print("=====================================================\n")

    top = tag_counts.head(top_n)
    # Plot top tags by in smaller communities
    plt.figure(figsize=(10, 6))
    plt.bar(top["tag"], top["count"])
    plt.xlabel("Meme keyword tag")
    plt.ylabel("Occurrences in smaller subreddits")
    plt.title(f"Keywords in smaller Reddit communities "
              f"(< {SMALL_SUBSCRIBERS_THRESHOLD:,} subscribers)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
