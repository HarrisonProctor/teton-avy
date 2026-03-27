import json
import csv
from pathlib import Path

def main():
    in_jsonl = Path("btac_teton_am_scrape/teton_am_archive.jsonl")
    out_csv = Path("btac_teton_am_scrape/teton_am_archive_clean.csv")
    out_jsonl = Path("btac_teton_am_scrape/teton_am_archive_clean.jsonl")

    fieldnames = [
        "date",
        "morning_temperature",
        "recent_snowfall",
        "wind_speed",
        "morning_avalanche_hazard"
    ]

    complete = []
    total = 0

    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if all(record.get(k) not in (None, "") for k in fieldnames):
                complete.append(record)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in complete:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(complete)

    print(f"Total: {total} | Kept: {len(complete)} | Removed: {total - len(complete)}")
    print(f"Saved to: {out_csv}, {out_jsonl}")

if __name__ == "__main__":
    main()
