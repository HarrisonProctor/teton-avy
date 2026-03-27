import asyncio
import csv
import json
import re
from datetime import date, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

START_DATE = date(2011, 1, 1)
END_DATE = date(2022, 12, 31)

ARCHIVE_URL = "https://archive.bridgertetonavalanchecenter.org/viewTeton"

OUT_DIR = Path("btac_teton_am_scrape")
OUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUT_DIR / "teton_am_archive.csv"
JSONL_PATH = OUT_DIR / "teton_am_archive.jsonl"

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def clean(s):
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_number(s):
    if not s:
        return None
    # Find the first occurrence of a number (supports negatives and decimals)
    # E.g., '13 º F' -> 13.0
    # E.g., '4"/ 0.40' -> 4.0
    m = re.search(r"(-?\d+\.?\d*)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def first_match(patterns, text, flags=re.I):
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return clean(m.group(1))
    return None

def extract_hazard_rating(rating_str):
    if not rating_str: return None
    # Look for explicit 1-5 number (e.g. "3-CONSIDERABLE")
    m = re.search(r"([1-5])", rating_str)
    if m: return int(m.group(1))
    
    r = rating_str.lower()
    if 'extreme' in r: return 5
    if 'high' in r: return 4
    if 'considerable' in r: return 3
    if 'moderate' in r: return 2
    if 'low' in r: return 1
    return None

def parse_from_text(text):
    t = text or ""
    data = {
        "morning_avalanche_hazard": None,
        "recent_snowfall": None,
        "wind_speed": None,
        "morning_temperature": None,
    }

    overall = first_match([
        r"overall avalanche danger[:\s]+(low|moderate|considerable|high|extreme)",
        r"avalanche danger[:\s]+(low|moderate|considerable|high|extreme)",
        r"danger rating[:\s]+(low|moderate|considerable|high|extreme)",
    ], t)
    data["morning_avalanche_hazard"] = extract_hazard_rating(overall)
    
    data["recent_snowfall"] = first_match([
        r"(?:recent snowfall|new snow|24[- ]?hour snowfall|overnight snowfall)[:\s]+([-\d\.]+\s*(?:in|inch|inches|cm))",
    ], t)

    data["wind_speed"] = first_match([
        r"(?:wind speed|average wind|winds?)[:\s]+([-\d\.]+\s*(?:mph|m/s|km/h|kts))",
    ], t)

    data["morning_temperature"] = first_match([
        r"(?:5 am temp|temperature|temp|air temperature)[:\s]+([-\d\.]+\s*(?:°f|°c|f|c))",
    ], t)

    return data

def enhance_data_from_tables(data, tables):
    hazard_str = None
    
    for tbl in tables:
        rows = tbl.get("rows", [])
        if not rows or len(rows) < 2:
            continue
            
        header = [str(r).lower() for r in rows[0]]
        header_text = " ".join(header)

        if "morning rating" in header_text or "hazard rating" in header_text:
            for row in rows[1:]:
                if len(row) >= 2:
                    loc = row[0].lower()
                    rating = row[1]
                    if "high" in loc and not hazard_str:
                        hazard_str = rating
                    if "mid" in loc and not hazard_str:
                        hazard_str = rating

        if "temp" in header_text or "wind" in header_text:
            temp_idx = -1
            wind_idx = -1
            for i, h in enumerate(header):
                if "5 am temp" in h and temp_idx == -1: temp_idx = i
                elif "temp" in h and temp_idx == -1: temp_idx = i
                if "avg. wind speed" in h or "avg wind speed" in h or "wind speed" in h: wind_idx = i
            
            first_data = rows[1]
            if temp_idx != -1 and temp_idx < len(first_data) and not data["morning_temperature"]:
                data["morning_temperature"] = first_data[temp_idx]
            if wind_idx != -1 and wind_idx < len(first_data) and not data["wind_speed"]:
                data["wind_speed"] = first_data[wind_idx]

        if "snowfall" in header_text or "new snow" in header_text:
            snow_idx = -1
            for i, h in enumerate(header):
                if "snowfall/prec." in h or "24hr snow" in h or "recent snowfall" in h or "new snow" in h or "snowfall" in h:
                    if "total" not in h:
                        snow_idx = i
            
            if snow_idx != -1 and not data["recent_snowfall"]:
                for r in rows[1:]:
                    if snow_idx < len(r) and r[snow_idx].strip():
                        data["recent_snowfall"] = r[snow_idx]
                        break

    if hazard_str and not data["morning_avalanche_hazard"]:
        data["morning_avalanche_hazard"] = extract_hazard_rating(hazard_str)

    return data

async def scrape_day(browser, sem, d):
    async with sem:
        context = await browser.new_context()
        page = await context.new_page()
        try:
            formatted_us = d.strftime("%m/%d/%Y")
            
            res = await context.request.post(ARCHIVE_URL, form={
                "forecast_date": formatted_us,
                "submit": "View"
            })
            html_text = await res.text()
            
            await page.set_content(html_text)
            
            body_text = clean(await page.locator("body").inner_text())
            tables = await page.locator("table").evaluate_all("""
                tables => tables.map((table, idx) => {
                    const rows = [...table.querySelectorAll("tr")].map(tr =>
                        [...tr.querySelectorAll("th,td")].map(td => td.innerText.trim())
                    );
                    return {table_index: idx, rows};
                })
            """)

            parsed = parse_from_text(body_text)
            parsed = enhance_data_from_tables(parsed, tables)

            # Clean all data points to only be pure numbers
            record = {
                "date": d.isoformat(),
                "morning_temperature": extract_number(parsed.get("morning_temperature")),
                "recent_snowfall": extract_number(parsed.get("recent_snowfall")),
                "wind_speed": extract_number(parsed.get("wind_speed")),
                "morning_avalanche_hazard": parsed.get("morning_avalanche_hazard")
            }
        except Exception as e:
            record = {
                "date": d.isoformat(),
                "morning_temperature": None,
                "recent_snowfall": None,
                "wind_speed": None,
                "morning_avalanche_hazard": None
            }
        finally:
            await context.close()
        return record

async def main():
    dates = list(daterange(START_DATE, END_DATE))
    total = len(dates)
    print(f"Scraping {total} days from {START_DATE} to {END_DATE}...")

    records = []
    sem = asyncio.Semaphore(50)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        tasks = [scrape_day(browser, sem, d) for d in dates]
        count = 0
        for future in asyncio.as_completed(tasks):
            res = await future
            records.append(res)
            count += 1
            if count % 100 == 0 or count == total:
                print(f"Progress: {count}/{total} pages scraped.")

        await browser.close()
        
    records.sort(key=lambda r: r["date"])

    fieldnames = [
        "date",
        "morning_temperature",
        "recent_snowfall",
        "wind_speed",
        "morning_avalanche_hazard"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} numerically cleaned records to {CSV_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
