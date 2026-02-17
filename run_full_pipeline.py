"""Run full pipeline: scrape (12 tokens) -> unique videos (latest batch) -> viral author filter."""
import os
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script, label, env=None):
    print(f"--- {label}: {script} ---")
    r = subprocess.run([sys.executable, script], cwd=SCRIPT_DIR, env=env)
    if r.returncode != 0:
        print(f"Pipeline failed at {script} (exit code {r.returncode}).")
        sys.exit(r.returncode)


def main():
    os.chdir(SCRIPT_DIR)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_output = f"tiktok_videos_unique_{timestamp}.jsonl"
    authors_output = f"authors_viral_filtered_{timestamp}.json"

    run_step("apifyAPI_connect.py", "Scraper (12 accounts)")

    env_unique = os.environ.copy()
    env_unique["MAKE_UNIQUE_OUTPUT_FILE"] = unique_output
    run_step("make_unique_videos_file.py", "Unique videos merge", env=env_unique)

    env_filter = os.environ.copy()
    env_filter["FILTER_AUTHORS_INPUT_FILE"] = os.path.join("dataset_50vid_of_prof", unique_output)
    env_filter["FILTER_AUTHORS_OUTPUT_FILE"] = os.path.join("dataset_50vid_of_prof", authors_output)
    run_step("filter_authors_viral.py", "Viral author filter", env=env_filter)

    print(f"Unique file: dataset_50vid_of_prof/{unique_output}")
    print(f"Authors file: dataset_50vid_of_prof/{authors_output}")
    print("--- Pipeline finished ---")


if __name__ == "__main__":
    main()
