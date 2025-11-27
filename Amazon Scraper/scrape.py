import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import argparse
from datetime import datetime
import os
import re
from tqdm import tqdm

BASE_SEARCH_URL = "https://www.amazon.in/s"
QUERY = "laptop"
HEADERS_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
]
REQUEST_TIMEOUT = 15
MIN_DELAY = 0.3
MAX_DELAY = 1.2

def random_delay():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def build_search_url(page=1):
    return f"{BASE_SEARCH_URL}?k={requests.utils.requote_uri(QUERY)}&page={page}"

def parse_price(card):
    offscreen = card.select_one('span.a-offscreen')
    if offscreen and offscreen.text.strip():
        val = re.sub(r'[^0-9]', '', offscreen.text)
        return val
    whole = card.select_one('span.a-price-whole')
    frac = card.select_one('span.a-price-fraction')
    if whole:
        value = whole.text.strip()
        if frac:
            value += frac.text.strip()
        value = re.sub(r'[^0-9]', '', value)
        return value
    return None

def parse_rating(card):
    r = card.select_one('span.a-icon-alt')
    if r:
        text = r.text.strip()
        m = re.search(r"([0-9]+\.?[0-9]?)", text)
        if m:
            return m.group(1) + "/5"
    return None

def is_sponsored(card):
    if card.find(string=lambda t: t and 'Sponsored' in t):
        return True
    if card.find(string=lambda t: t and (t.strip() == 'Ad' or 'ADVERTISEMENT' in t.upper())):
        return True
    return False

def extract_image_url(card):
    img = card.select_one('img.s-image')
    if img:
        for attr in ('data-src', 'src', 'data-old-hires'):
            u = img.get(attr)
            if u:
                return u
    return None

def extract_title(card):
    h2 = card.select_one('h2 a')
    if h2 and h2.get_text(strip=True):
        return h2.get_text(strip=True)
    img = card.select_one('img.s-image')
    if img and img.get('alt'):
        return img.get('alt').strip()
    span1 = card.select_one('span.a-size-medium')
    if span1 and span1.get_text(strip=True):
        return span1.get_text(strip=True)
    span2 = card.select_one('span.a-size-base-plus')
    if span2 and span2.get_text(strip=True):
        return span2.get_text(strip=True)
    return None

def extract_product_url_and_asin(card):
    asin = card.get('data-asin') or None
    url = None
    a = card.select_one("a.a-link-normal.s-no-outline")
    if not a:
        a = card.select_one("a.a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal")
    if not a:
        a = card.select_one("h2 a")
    if a and a.get('href'):
        href = a['href'].strip()
        if not href.startswith('http'):
            href = requests.compat.urljoin("https://www.amazon.in", href)
        href = href.split('ref=')[0]
        url = href
        if not asin:
            m = re.search(r"/dp/([A-Z0-9]{10})", href)
            if m:
                asin = m.group(1)
    return url, asin


def scrape_page(html_text):
    soup = BeautifulSoup(html_text, 'lxml')
    results = []
    cards = soup.select("div.s-result-item[data-asin]")
    for card in cards:
        asin = card.get('data-asin')
        if not asin:
            continue
        title = extract_title(card)
        image_url = extract_image_url(card)
        rating = parse_rating(card)
        price = parse_price(card)
        product_url, asin_parsed = extract_product_url_and_asin(card)
        result_type = 'Ad' if is_sponsored(card) else 'Organic'
        results.append({
            'asin': asin_parsed or asin,
            'product_name': title,
            'image_url': image_url,
            'rating': rating,
            'price': price,
            'result_type': result_type,
            'product_url': product_url,
        })
    return results

def fetch_search_page(page=1, use_selenium=False):
    url = build_search_url(page)
    headers = {"User-Agent": random.choice(HEADERS_LIST), 'Accept-Language': 'en-IN,en;q=0.9'}
    if use_selenium:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.options import Options
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'user-agent={headers["User-Agent"]}')
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            driver.get(url)
            time.sleep(2)
            html = driver.page_source
            driver.quit()
            return html
        except:
            return None
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.text
        return None
    except:
        return None

def detect_total_pages(html):
    soup = BeautifulSoup(html, 'lxml')
    nums = soup.select('span.s-pagination-item')
    pages = []
    for n in nums:
        t = n.get_text(strip=True)
        if t.isdigit():
            pages.append(int(t))
    if pages:
        return max(pages)
    return 1

def main(pages=5, use_selenium=False, out_dir='output'):
    os.makedirs(out_dir, exist_ok=True)
    first_html = fetch_search_page(1, use_selenium)
    if not first_html:
        first_html = fetch_search_page(1, True)
        if not first_html:
            return
    total_detected = detect_total_pages(first_html)
    if not pages or pages > total_detected:
        pages = total_detected
    all_results = []
    for p in tqdm(range(1, pages + 1), desc="Fetching pages"):
        html = fetch_search_page(page=p, use_selenium=use_selenium)
        if not html:
            html = fetch_search_page(page=p, use_selenium=True)
            if not html:
                break
        page_results = scrape_page(html)
        if not page_results:
            break
        all_results.extend(page_results)
        random_delay()
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'{timestamp}.csv')
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=None, help='Number of pages to scrape')
    parser.add_argument('--selenium', action='store_true', help='Force use of Selenium')
    parser.add_argument('--out', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    main(pages=args.pages, use_selenium=args.selenium, out_dir=args.out)