import pandas as pd
import numpy as np
from tqdm import tqdm
from selenium.common.exceptions import WebDriverException

from config import (
    SUBREDDIT,
    TIMEFRAME,
    LIMIT,
    OUTPUT_XLSX,
    VIRAL_PERCENTILE,
)
from reddit_utils import fetch_top_posts, build_rows
from reverse_search_and_analysis import (
    make_driver,
    google_reverse_image_exact,
    summarize_size_tendencies,
    write_top_sheet,
    append_reverse_sheet,
    analyze_meme_popularity,
    analyze_small_community_keywords,
)


def main():
    print(f"Fetching r/{SUBREDDIT} top posts ({TIMEFRAME}, limit={LIMIT})…")
    posts = fetch_top_posts()
    rows = build_rows(posts)
    if not rows:
        print("No image posts found.")
        return

    df = pd.DataFrame(rows)
    cutoff = np.percentile(df["hot"], VIRAL_PERCENTILE)
    df["viral"] = np.where(df["hot"] >= cutoff, "YES", "NO")

    df_out = df[[
        "post link",
        "image link",
        "# of upvotes",
        "# of comments",
        "viral",
        "image keywords",
        "image keyword scores",
    ]].copy()

    print("Starting Google reverse image search")
    driver = None

    try:
        driver = make_driver()
        reverse_rows = []

        for image_link in tqdm(df_out["image link"].tolist(), desc="Reverse searching"):
            try:
                results = google_reverse_image_exact(image_link, driver)
            except WebDriverException as e:
                print(f"[warn] Selenium error for {image_link}: {e}")
                results = []
            except Exception as e:
                print(f"[warn] Reverse search failed for {image_link}: {e}")
                results = []

            if not results:
                reverse_rows.append({
                    "original image link": image_link,
                    "result url": "",
                    "community about": "No exact matches found",
                    "page title": "",
                    "subreddit": "",
                    "subreddit subscribers": None,
                    "community size bucket": "",
                    "reddit community about": "",
                    "match_image_link": "",
                    "match_image_keywords": "",
                })
                from config import sleep
                import random as _rnd
                sleep(1.0 + _rnd.uniform(0.0, 0.8))
                continue

            for r in results:
                reverse_rows.append({
                    "original image link": image_link,
                    "result url": r.get("result_url", ""),
                    "community about": r.get("community_about", "Website"),
                    "page title": r.get("page_title", ""),
                    "subreddit": r.get("subreddit", ""),
                    "subreddit subscribers": r.get("subreddit subscribers", None),
                    "community size bucket": r.get("community size bucket", ""),
                    "reddit community about": r.get("reddit community about", ""),
                    "match_image_link": r.get("match_image_link", ""),
                    "match_image_keywords": r.get("match_image_keywords", ""),
                })

            from config import sleep
            import random as _rnd
            sleep(2.5 + _rnd.uniform(0.0, 1.2))

        tendency_map = summarize_size_tendencies(reverse_rows)
        df_out["where similar posts appear (community size)"] = df_out["image link"].map(
            tendency_map
        ).fillna("unknown")

        write_top_sheet(df_out, OUTPUT_XLSX)
        append_reverse_sheet(reverse_rows, OUTPUT_XLSX)

        analyze_meme_popularity(df_out)
        analyze_small_community_keywords(reverse_rows)

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    print(f"Wrote {len(df_out)} image posts to {OUTPUT_XLSX}")
    print(
        "Sheet 'reverse_search' with links, subreddit, subscribers, "
        "size, category labels, and keywords for matched Reddit images"
    )
    


if __name__ == "__main__":
    main()
