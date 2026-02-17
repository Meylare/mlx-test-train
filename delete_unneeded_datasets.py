"""Delete dataset files that are no longer needed (content already in selected outputs)."""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_50VID = os.path.join(SCRIPT_DIR, "dataset_50vid_of_prof")
DATASET = os.path.join(SCRIPT_DIR, "dataset")

# Keep only these in dataset_50vid_of_prof
KEEP_50VID = {"tiktok_videos_unique.jsonl", "authors_viral_filtered.json"}

deleted = 0
for fn in os.listdir(DATASET_50VID):
    path = os.path.join(DATASET_50VID, fn)
    if not os.path.isfile(path):
        continue
    if fn in KEEP_50VID:
        continue
    os.remove(path)
    print("Deleted:", path)
    deleted += 1

if os.path.isdir(DATASET):
    for fn in os.listdir(DATASET):
        path = os.path.join(DATASET, fn)
        if os.path.isfile(path):
            os.remove(path)
            print("Deleted:", path)
            deleted += 1

print("Done. Deleted", deleted, "files.")
print("Kept: tiktok_videos_unique.jsonl, authors_viral_filtered.json")
