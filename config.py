
import math
import time

REDDIT_BASE = "https://www.reddit.com"
SUBREDDIT = "memes"
LIMIT = 50
TIMEFRAME = "day"

#output excel file for results
OUTPUT_XLSX = "memes_top5.xlsx"

HEADERS = {
    "User-Agent": "script:memes-top5: (by u/rami)"
}

#fixed epoch reddit hot algorithm
REDDIT_EPOCH = 1134028003

VIRAL_PERCENTILE = 80
MAX_IMAGES = 5
SMALL_SUBSCRIBERS_THRESHOLD = 100_000

GOOGLE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SELENIUM_HEADLESS = False
SELENIUM_PAGELOAD_TIMEOUT = 45
SELENIUM_WAIT_S = 3.5
MAX_RESULTS_PER_IMAGE = 10

CHROME_USER_DATA_DIR = r""
CHROME_PROFILE_DIR = ""


def sleep(t: float) -> None:
    time.sleep(t)

#computes hot score for a reddit post 
def hot_score_from_post(post: dict) -> float:
    score = post.get("score", post.get("ups", 0))
    created = post.get("created_utc", 0)
    s = 1 if score > 0 else (-1 if score < 0 else 0)
    order = math.log10(max(abs(score), 1))
    seconds = created - REDDIT_EPOCH
    return order + s * (seconds / 45000.0)
