import json
import os
from datetime import datetime

import requests
from apify_client import ApifyClient
from apify_client.errors import ApifyApiError

TOKENS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.txt")
try:
    with open(TOKENS_FILE, encoding="utf-8") as f:
        TOKENS = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    raise SystemExit(f"Tokens file not found: {TOKENS_FILE}. Create it with one Apify token per line.")
if not TOKENS:
    raise SystemExit(f"No tokens found in {TOKENS_FILE}. Add one Apify token per line.")
THRESHOLD = 4.80  # Limit usage to $4.80 per account to be safe

def get_current_usage(token):
    """Checks monthly usage in USD via the Apify REST API."""
    url = "https://api.apify.com/v2/users/me/usage/monthly"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # 'baseAmountUsd' reflects the total USD spent in the current cycle
        return data.get("data", {}).get("baseAmountUsd", 0)
    return float('inf')  # If error, treat as exhausted

def get_client_for_account(account_index):
    """Return (ApifyClient, account_index) for the given account if usage < THRESHOLD, else (None, account_index)."""
    n = len(TOKENS)
    idx = account_index % n
    token = TOKENS[idx]
    usage = get_current_usage(token)
    print(f"Account {idx + 1}/{n} ({token[:12]}...) usage: ${usage}")
    if usage < THRESHOLD:
        return ApifyClient(token), idx
    return None, idx

# Load profiles from file and keep up to MAX_PROFILES usernames.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_50vid_of_prof")
PROFILES_FILE = os.path.join(SCRIPT_DIR, "unique_authors.txt")
PROFILE_OFFSET_FILE = os.path.join(SCRIPT_DIR, "apify_profile_offset.txt")
ATTEMPTED_PROFILES_FILE = os.path.join(SCRIPT_DIR, "apify_attempted_profiles.txt")
MAX_PROFILES = 2000


def _normalize_username(s):
    """Canonical form for comparing usernames (strip and remove leading @)."""
    if not s:
        return ""
    return (s.strip() or "").lstrip("@").lower()


def _load_attempted_profiles(path):
    attempted = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                name = _normalize_username(line)
                if name:
                    attempted.add(name)
    except FileNotFoundError:
        pass
    return attempted


def _save_attempted_profiles(path, attempted):
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted(attempted):
            f.write(name + "\n")


def _collect_existing_authors(dataset_dir):
    """Collect author usernames already present in dataset JSONL/JSON files (so we don't re-scrape them)."""
    seen = set()
    if not os.path.isdir(dataset_dir):
        return seen
    # All tiktok_videos_*.jsonl (account files + unique)
    for fn in os.listdir(dataset_dir):
        if not fn.endswith(".jsonl") or not fn.startswith("tiktok_videos_"):
            continue
        path = os.path.join(dataset_dir, fn)
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        am = row.get("authorMeta") or {}
                        name = am.get("name") or ""
                        if name:
                            seen.add(_normalize_username(name))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
    # authors_viral_filtered.json
    af_path = os.path.join(dataset_dir, "authors_viral_filtered.json")
    if os.path.isfile(af_path):
        try:
            with open(af_path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                author = item.get("author") or {}
                name = author.get("name") or ""
                if name:
                    seen.add(_normalize_username(name))
        except (OSError, json.JSONDecodeError):
            pass
    return seen


with open(PROFILES_FILE, encoding="utf-8") as f:
    _all_profiles = [line.strip() for line in f if line.strip()]
_pool = _all_profiles[:MAX_PROFILES]

if not _pool:
    raise SystemExit(
        f"No profiles to scrape. File has {len(_all_profiles)} lines."
        f" Add usernames to {PROFILES_FILE}."
    )

# Persist offset so each run uses the next chunk of usernames (fewer repeats across runs).
try:
    with open(PROFILE_OFFSET_FILE, encoding="utf-8") as f:
        _start = int(f.read().strip())
except (FileNotFoundError, ValueError):
    _start = 0
pool_size = len(_pool)
# This run: use pool starting at _start (wrapping), so we get a fresh rotation each run.
_indices = [(_start + i) % pool_size for i in range(pool_size)]
_profiles = [_pool[i] for i in _indices]
_rotated_profiles = list(_profiles)
# Advance offset by one account's worth so next run gets different usernames.
step = max(1, pool_size // len(TOKENS))
_next_start = (_start + step) % pool_size
with open(PROFILE_OFFSET_FILE, "w", encoding="utf-8") as f:
    f.write(str(_next_start))
if _start != 0:
    print(f"Profile offset: {_start} (next run starts at index {_next_start}).")
if _next_start < _start or (_start == 0 and _next_start == step and step < pool_size):
    print("Next run will use a different slice of usernames.")

# Exclude authors we already have in the dataset (no re-scraping).
existing_authors = _collect_existing_authors(DATASET_DIR)
_profiles = [p for p in _profiles if _normalize_username(p) not in existing_authors]
if existing_authors:
    print(f"Excluding {len(existing_authors)} authors already in dataset.")

# Exclude profiles already attempted in previous runs so each run advances.
attempted_profiles = _load_attempted_profiles(ATTEMPTED_PROFILES_FILE)
_profiles = [p for p in _profiles if _normalize_username(p) not in attempted_profiles]
if attempted_profiles:
    print(f"Excluding {len(attempted_profiles)} usernames already attempted in previous runs.")

# If everything has been attempted already, reset attempts and continue from remaining pool.
if not _profiles and attempted_profiles:
    attempted_profiles = set()
    _save_attempted_profiles(ATTEMPTED_PROFILES_FILE, attempted_profiles)
    _profiles = [p for p in _rotated_profiles if _normalize_username(p) not in existing_authors]
    print("Attempted-profile history exhausted; resetting attempt history.")

if not _profiles:
    raise SystemExit(
        "No profiles left after excluding authors already in dataset. "
        "Add new usernames to unique_authors.txt or remove/rename existing dataset files to re-scrape."
    )

n_tokens = len(TOKENS)
# Split profiles so each account gets a different slice (minimize overlap).
def _slice_profiles_for_account(account_index):
    L, n = len(_profiles), n_tokens
    start = (account_index * L) // n
    end = ((account_index + 1) * L) // n
    return _profiles[start:end]

profile_chunks = [_slice_profiles_for_account(i) for i in range(n_tokens)]
print(f"Using {len(_profiles)} profiles total, split across {n_tokens} accounts (~{len(_profiles) // max(1, n_tokens)} each).")

# Input for free-tiktok-scraper: must have at least one of profiles, hashtags,
# searchQueries, postURLs, or music (see actor input schema on Apify).
# resultsPerPage = max videos per profile (and per hashtag/search query).
VIDEOS_PER_PROFILE = 50

def run_input_for_account(account_index):
    return {
        "profiles": profile_chunks[account_index],
        "resultsPerPage": VIDEOS_PER_PROFILE,
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
    }

# Output: one file per account in dataset_50vid_of_prof
os.makedirs(DATASET_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
saved_count = 0

# Run the scraper once per account; each account gets its own slice of profiles.
for account_index in range(n_tokens):
    account_num = account_index + 1
    client, idx = get_client_for_account(account_index)
    if client is None:
        print(f"Skipping account {account_num}/{n_tokens} (over usage threshold).")
        continue
    run_input = run_input_for_account(account_index)
    if not run_input["profiles"]:
        print(f"Skipping account {account_num}/{n_tokens} (no new profiles for this slice).")
        continue
    print(f"Account {account_num}: {len(run_input['profiles'])} profiles.")
    try:
        print(f"Running scraper for account {account_num}/{n_tokens}...")
        run = client.actor('clockworks/free-tiktok-scraper').call(run_input=run_input)
        attempted_profiles.update(
            _normalize_username(p) for p in run_input["profiles"] if _normalize_username(p)
        )
        _save_attempted_profiles(ATTEMPTED_PROFILES_FILE, attempted_profiles)
    except ApifyApiError as e:
        err_msg = str(e).lower()
        if "exceed" in err_msg and ("remaining" in err_msg or "usage" in err_msg):
            print(f"Account {account_num}/{n_tokens} has insufficient remaining credits, skipping.")
            continue
        raise

    dataset_id = run["defaultDatasetId"]
    OUTPUT_FILENAME = f"tiktok_videos_account{account_num}_{timestamp}.jsonl"
    OUTPUT_PATH = os.path.join(DATASET_DIR, OUTPUT_FILENAME)
    print(f"Account {account_num}/{n_tokens} finished. Fetching dataset -> {OUTPUT_PATH}")
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in client.dataset(dataset_id).iterate_items():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    saved_count += count
    print(f"Saved {count} videos to {OUTPUT_PATH}")

print(f"Done. {saved_count} videos total across {n_tokens} accounts.")
