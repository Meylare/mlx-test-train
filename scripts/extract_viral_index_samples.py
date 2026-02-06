import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

DEFAULT_ID_CANDIDATES = ("id", "video_id", "videoid", "vid")
DEFAULT_TRANSCRIPT_CANDIDATES = (
    "transcript",
    "transcription",
    "asr_text",
    "asr",
    "text",
)


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_jsonl(input_path: Path) -> list[dict]:
    videos = []
    skipped = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "id" not in obj:
                skipped += 1
                continue
            vi = obj.get("targets", {}).get("viral_index")
            if vi is None:
                skipped += 1
                continue
            videos.append(obj)
    if skipped:
        print(f"Пропущено записей без id/viral_index: {skipped}")
    return videos


def select_column(
    available: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    available_lower = {name.lower(): name for name in available}
    for candidate in candidates:
        actual = available_lower.get(candidate.lower())
        if actual:
            return actual
    return None


def load_transcripts(
    parquet_path: Path,
    candidate_ids: set[str],
    id_col: str | None,
    transcript_col: str | None,
) -> tuple[dict[str, str], str, str]:
    parquet_file = pq.ParquetFile(parquet_path)
    columns = parquet_file.schema_arrow.names

    if id_col is None:
        id_col = select_column(columns, DEFAULT_ID_CANDIDATES)
    if transcript_col is None:
        transcript_col = select_column(columns, DEFAULT_TRANSCRIPT_CANDIDATES)

    if id_col is None or transcript_col is None:
        missing = []
        if id_col is None:
            missing.append("id column")
        if transcript_col is None:
            missing.append("transcript column")
        missing_text = ", ".join(missing)
        available_text = ", ".join(columns)
        raise ValueError(
            f"Не удалось определить {missing_text} в parquet. "
            f"Доступные колонки: {available_text}. "
            "Передайте --parquet-id-col и/или --parquet-transcript-col."
        )

    transcripts: dict[str, str] = {}
    for batch in parquet_file.iter_batches(columns=[id_col, transcript_col]):
        data = batch.to_pydict()
        ids = data.get(id_col, [])
        texts = data.get(transcript_col, [])
        for item_id, text in zip(ids, texts, strict=False):
            if item_id is None:
                continue
            item_id = str(item_id)
            if item_id not in candidate_ids:
                continue
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            transcripts.setdefault(item_id, text)
    return transcripts, id_col, transcript_col


def sample_pool(
    pool: list[dict],
    desired: int,
    selected_ids: set[str],
    label: str,
) -> list[dict]:
    available = [v for v in pool if v["id"] not in selected_ids]
    if len(available) < desired:
        print(f"Внимание: в пуле {label} только {len(available)} видео, взяты все")
        desired = len(available)
    if desired == 0:
        return []
    picks = random.sample(available, desired)
    for video in picks:
        selected_ids.add(video["id"])
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description="Выборка по Viral Index + транскрипции")
    parser.add_argument("--input", default="dataset_train.jsonl", help="Путь к JSONL")
    parser.add_argument(
        "--output",
        default="dataset_train_vi_samples_with_transcript.jsonl",
        help="Путь к выходному JSONL",
    )
    parser.add_argument("--parquet", default="train_stage2.parquet", help="Путь к parquet")
    parser.add_argument("--seed", type=int, default=42, help="Seed для случайности")
    parser.add_argument("--top-k", type=int, default=2000, help="Размер топ пула")
    parser.add_argument("--high-n", type=int, default=300, help="Сэмпл из top-k по VI")
    parser.add_argument("--low-n", type=int, default=300, help="Сэмпл из bottom-k по VI")
    parser.add_argument("--mid-min", type=float, default=-1.0, help="Нижняя граница VI")
    parser.add_argument("--mid-max", type=float, default=1.0, help="Верхняя граница VI")
    parser.add_argument("--mid-n", type=int, default=400, help="Сэмпл из среднего диапазона")
    parser.add_argument("--parquet-id-col", default=None, help="Колонка id в parquet")
    parser.add_argument(
        "--parquet-transcript-col",
        default=None,
        help="Колонка транскрипции в parquet",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = resolve_path(project_root, args.input)
    output_path = resolve_path(project_root, args.output)
    parquet_path = resolve_path(project_root, args.parquet)

    random.seed(args.seed)

    print(f"Читаю JSONL: {input_path}")
    videos = load_jsonl(input_path)
    print(f"Всего видео: {len(videos)}")

    print("Сортирую по Viral Index...")
    videos_sorted = sorted(videos, key=lambda x: x["targets"]["viral_index"], reverse=True)

    top_k = max(args.top_k, 1)
    high_pool = videos_sorted[:top_k]
    low_pool = videos_sorted[-top_k:]
    mid_pool = [
        v
        for v in videos
        if args.mid_min <= v["targets"]["viral_index"] <= args.mid_max
    ]

    candidate_ids = {v["id"] for v in high_pool + low_pool + mid_pool}

    print(f"Читаю транскрипции из parquet: {parquet_path}")
    transcripts, id_col, transcript_col = load_transcripts(
        parquet_path,
        candidate_ids,
        args.parquet_id_col,
        args.parquet_transcript_col,
    )
    print(
        "Транскрипции загружены: "
        f"{len(transcripts)} (id_col={id_col}, transcript_col={transcript_col})"
    )

    high_pool = [v for v in high_pool if v["id"] in transcripts]
    low_pool = [v for v in low_pool if v["id"] in transcripts]
    mid_pool = [v for v in mid_pool if v["id"] in transcripts]

    print(f"High pool (top-{top_k}) с транскрипцией: {len(high_pool)}")
    print(f"Low pool (bottom-{top_k}) с транскрипцией: {len(low_pool)}")
    print(
        f"Middle pool VI in [{args.mid_min}, {args.mid_max}] с транскрипцией: {len(mid_pool)}"
    )

    selected_ids: set[str] = set()
    high_samples = sample_pool(high_pool, args.high_n, selected_ids, "high")
    low_samples = sample_pool(low_pool, args.low_n, selected_ids, "low")
    mid_samples = sample_pool(mid_pool, args.mid_n, selected_ids, "middle")

    selected_videos = high_samples + low_samples + mid_samples

    for video in selected_videos:
        video["transcript"] = transcripts.get(video["id"], "")

    missing_transcripts = [v for v in selected_videos if not v.get("transcript")]
    if missing_transcripts:
        print(
            "Внимание: "
            f"{len(missing_transcripts)} записей без транскрипции после объединения."
        )

    print(f"Всего уникальных видео после объединения: {len(selected_videos)}")
    print(f"Сохраняю в {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for video in selected_videos:
            f.write(json.dumps(video, ensure_ascii=False) + "\n")

    print("Готово!")
    print(f"Создан файл: {output_path}")
    print(f"Итого записей: {len(selected_videos)}")


if __name__ == "__main__":
    main()
