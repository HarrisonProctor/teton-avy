import json
import csv
from pathlib import Path

def print_summary(total, kept):
    print(f"Total entries parsed: {total}")
    print(f"Fully complete entries kept: {kept}")
    print(f"Incomplete entries removed: {total - kept}")

def main():
    IN_JSONL = Path("btac_teton_am_scrape/teton_am_archive.jsonl")
    
    OUT_CSV = Path("btac_teton_am_scrape/teton_am_archive_clean.csv")
    OUT_JSONL = Path("btac_teton_am_scrape/teton_am_archive_clean.jsonl")
    
    # We will use the original fieldnames from the jsonl structure
    fieldnames = [
        "date",
        "morning_temperature",
        "recent_snowfall",
        "wind_speed",
        "morning_avalanche_hazard"
    ]
    
    complete_records = []
    total_records = 0
    
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            record = json.loads(line)
            
            # Check if any value in the dict is None or an empty string
            is_complete = True
            for field in fieldnames:
                val = record.get(field)
                if val is None or val == "":
                    is_complete = False
                    break
                    
            if is_complete:
                complete_records.append(record)
                
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in complete_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(complete_records)
        
    print_summary(total_records, len(complete_records))
    print(f"Saved cleanly to:\n- {OUT_CSV}\n- {OUT_JSONL}")

if __name__ == "__main__":
    main()
