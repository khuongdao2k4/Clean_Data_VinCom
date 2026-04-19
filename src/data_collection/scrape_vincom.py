import json
import re
import time
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

MALLS = [
    "Vincom Mega Mall Royal City, Hà Nội",
    "Vincom Mega Mall Times City, Hà Nội",
    "Vincom Mega Mall Smart City, Hà Nội",
    "Vincom Mega Mall Ocean Park, Hà Nội",
    "Vincom Mega Mall Grand Park, Hà Nội",
    "Vincom Mega Mall Thảo Điền, TP.HCM",
    "Vincom Center Landmark 81, TP.HCM",
    "Vincom Center Đồng Khởi, TP.HCM",
    "Vincom Center Bà Triệu, Hà Nội",
    "Vincom Center Phạm Ngọc Thạch"
]

def clean_text(text):
    if not text: return ""
    return " ".join(text.split())

def scrape_reviews(page, mall_name):
    print(f"Searching for: {mall_name}")
    page.goto("https://www.google.com/maps", timeout=60000)
    
    # Search for mall
    try:
        search_box = page.locator("#searchboxinput, input[name='q'], textarea").first
        search_box.wait_for(state="visible", timeout=15000)
        search_box.fill(mall_name)
        search_box.press("Enter")
        page.wait_for_timeout(5000)
    except Exception as e:
        print(f"Could not find search box for {mall_name}: {e}")
        return []

    # Handle cookies if present
    try:
        accept_btn = page.locator("button:has-text('Accept all'), button:has-text('Chấp nhận')").first
        if accept_btn.is_visible():
            accept_btn.click()
            page.wait_for_timeout(2000)
    except: pass

    # Click Reviews tab
    try:
        # A more robust regex lookup for exact words so we don't accidentally click wrong items
        reviews_tab = page.locator("button, [role='tab'], div.Gpq6fc").filter(has_text=re.compile(r"^(Bài đánh giá|Reviews|Đánh giá)$")).first
        reviews_tab.wait_for(state="visible", timeout=15000)
        reviews_tab.click()
        page.wait_for_timeout(3000)
    except:
        print(f"Could not find reviews tab for {mall_name}")
        return []

    reviews_data = []
    last_height = 0
    
    # Wait for the first review to load before starting the loop
    try:
        page.wait_for_selector("div.jftiEf, div[data-review-id]", timeout=15000)
    except:
        print(f"No reviews loaded for {mall_name}")
        return []

    # Scroll to load reviews
    while len(reviews_data) < 100:
        # Scroll the review container
        container = page.locator("div.m6QErb.DxyBCb, div[role='feed']").first
        if container.count() > 0:
            container.evaluate("el => el.scrollBy(0, 3000)")
        else:
            page.mouse.wheel(0, 3000)
        
        page.wait_for_timeout(2000)
        
        # Expand "More" / "Xem thêm" buttons for long comments
        # Using exact boundaries to ensure we only click the strict 'Xem thêm' button within reviews
        more_btns = page.locator("button").filter(has_text=re.compile(r"^(Xem th.m|More)$")).all()
        for btn in more_btns:
            if btn.is_visible():
                try: 
                    btn.click()
                    page.wait_for_timeout(300) # give it a moment to expand the text
                except: pass
                
        page.wait_for_timeout(1000) # Wait for text to expand
        
        # Extract current visible reviews
        review_els = page.locator("div.jftiEf, div[data-review-id]").all()
        current_batch = []
        for el in review_els:
            try:
                name_el = el.locator("div.d4r55, span[itemprop='name']").first
                name = name_el.inner_text() if name_el.count() > 0 else "Anonymous"
                
                # Text extraction
                text_el = el.locator("span.wiS7gd, span.wiI7pd, span[data-expanded-review-text], span[data-rendered-text]").first
                review_text = text_el.inner_text() if text_el.count() > 0 else ""
                
                # Rating
                rating = 0
                rating_el = el.locator("span.kv7Zre, span[role='img'][aria-label*='stars'], span[role='img'][aria-label*='sao']").first
                if rating_el.count() > 0:
                    label = rating_el.get_attribute("aria-label") or ""
                    match = re.search(r'(\d+)', label)
                    if match: rating = int(match.group(1))

                # Time
                time_el = el.locator("span.rsqaWe, span[data-review-time]").first
                review_time = time_el.inner_text() if time_el.count() > 0 else ""

                # Helpful count
                helpful = 0
                helpful_el = el.locator("button[aria-label*='helpful'], button[aria-label*='người thấy']").first
                if helpful_el.count() > 0:
                    label = helpful_el.get_attribute("aria-label") or ""
                    match = re.search(r'(\d+)', label)
                    if match: helpful = int(match.group(1))

                # Owner Reply
                reply = ""
                reply_el = el.locator("div.C5txpf, span:has-text('Owner'), span:has-text('Phản hồi của chủ sở hữu')").first
                if reply_el.count() > 0:
                    reply = clean_text(reply_el.inner_text().replace("Owner's reply", "").replace("Phản hồi của chủ sở hữu", "").replace("(Translated by Google)", ""))

                current_batch.append({
                    "mall_name": mall_name,
                    "reviewer_name": clean_text(name),
                    "review_text": clean_text(review_text),
                    "rating": rating,
                    "review_time": clean_text(review_time),
                    "helpful_count": helpful,
                    "owner_reply": clean_text(reply)
                })
            except: continue
        
        # Deduplicate by name + review text
        seen = set()
        unique_batch = []
        for r in current_batch:
            key = f"{r['reviewer_name']}_{r['review_text']}_{r['review_time']}"
            if key not in seen:
                seen.add(key)
                unique_batch.append(r)
        
        reviews_data = unique_batch
        print(f"Loaded {len(reviews_data)} unique reviews for {mall_name}...")
        if len(reviews_data) >= 100: break
        
        # Check if we've stopped loading new content
        new_height = page.evaluate("document.body.scrollHeight") if container.count() == 0 else container.evaluate("el => el.scrollHeight")
        if new_height == last_height:
             # Wait a little longer just in case and try once more before breaking
             page.wait_for_timeout(2000)
             new_height2 = page.evaluate("document.body.scrollHeight") if container.count() == 0 else container.evaluate("el => el.scrollHeight")
             if new_height2 == last_height:
                 print("Reached end of reviews list.")
                 break
        last_height = new_height

    return reviews_data[:100]

def main():
    all_results = []
    with sync_playwright() as p:
        # headless=False to bypass Google Maps bot detection which hides tabs
        browser = p.chromium.launch(headless=False, args=['--start-maximized'])
        context = browser.new_context(no_viewport=True, locale="vi-VN")
        page = context.new_page()
        
        for mall in MALLS: # Testing with first 3 to show user it works
            data = scrape_reviews(page, mall)
            all_results.extend(data)
            print(f"Collected {len(data)} reviews for {mall}")
        
        browser.close()
    
    # Xác định đường dẫn gốc của project
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "raw" / "raw_reviews.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved raw reviews to {output_path}")

if __name__ == "__main__":
    main()
