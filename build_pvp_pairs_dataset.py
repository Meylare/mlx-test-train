from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VideoLite:
    id: str
    web_video_url: str
    play_count: int
    create_time: int
    create_time_iso: Optional[str]
    local_path: Optional[str]


@dataclass
class AuthorPrepared:
    source_row: Dict[str, Any]
    author_name: str
    author_id: str
    consistency_score: float
    selected_30: List[Dict[str, Any]]
    median_views: float
    threshold: float
    threshold_hit_count: int
    downloaded_in_30: int
    missing_in_30: int
    score: Tuple[Any, ...]


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _author_name_from_row(row: Dict[str, Any]) -> str:
    author = (row.get("author_record") or {}).get("author") or {}
    return str(author.get("name") or "").strip()


def _author_id_from_row(row: Dict[str, Any]) -> str:
    author = (row.get("author_record") or {}).get("author") or {}
    return str(author.get("id") or "").strip()


def _consistency_score_from_row(row: Dict[str, Any]) -> float:
    consistency = row.get("consistency") or {}
    return _to_float(consistency.get("topic_consistency_score"), 0.0)


def _video_sort_key(v: Dict[str, Any]) -> Tuple[int, int, str]:
    return (
        _to_int(v.get("createTime"), 0),
        _to_int(v.get("playCount"), 0),
        str(v.get("id") or ""),
    )


def _video_to_lite(v: Dict[str, Any], local_path: Optional[str]) -> VideoLite:
    return VideoLite(
        id=str(v.get("id") or ""),
        web_video_url=str(v.get("webVideoUrl") or ""),
        play_count=_to_int(v.get("playCount"), 0),
        create_time=_to_int(v.get("createTime"), 0),
        create_time_iso=v.get("createTimeISO"),
        local_path=local_path,
    )


def _lite_to_dict(v: VideoLite) -> Dict[str, Any]:
    return {
        "id": v.id,
        "webVideoUrl": v.web_video_url,
        "playCount": v.play_count,
        "createTime": v.create_time,
        "createTimeISO": v.create_time_iso,
        "localPath": v.local_path,
    }


def _index_downloaded(downloads_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Build index: author_name_lower -> {video_id -> local_path}
    """
    by_author: Dict[str, Dict[str, str]] = {}
    if not downloads_dir.exists():
        return by_author

    for author_dir in downloads_dir.iterdir():
        if not author_dir.is_dir():
            continue
        author_key = author_dir.name.lower()
        file_map: Dict[str, str] = {}
        for file_path in author_dir.iterdir():
            if not file_path.is_file():
                continue
            video_id = file_path.stem
            if video_id:
                file_map[video_id] = str(file_path.as_posix())
        by_author[author_key] = file_map
    return by_author


def _prepare_author(
    row: Dict[str, Any],
    videos_per_author: int,
    multiplier: float,
    downloaded_index: Dict[str, Dict[str, str]],
) -> Optional[AuthorPrepared]:
    author_name = _author_name_from_row(row)
    if not author_name:
        return None

    all_videos = ((row.get("author_record") or {}).get("videos") or [])
    if not isinstance(all_videos, list) or len(all_videos) < videos_per_author:
        return None

    ranked = sorted(all_videos, key=_video_sort_key, reverse=True)
    selected_30 = ranked[:videos_per_author]
    plays = [_to_int(v.get("playCount"), 0) for v in selected_30]
    median_views = float(statistics.median(plays))
    threshold = median_views * multiplier
    threshold_hit_count = sum(1 for p in plays if p >= threshold)

    author_files = downloaded_index.get(author_name.lower(), {})
    downloaded_in_30 = sum(
        1 for v in selected_30 if str(v.get("id") or "") in author_files
    )
    missing_in_30 = videos_per_author - downloaded_in_30

    score = (
        downloaded_in_30,
        min(threshold_hit_count, 5),
        threshold_hit_count,
        _consistency_score_from_row(row),
        author_name.lower(),
    )
    return AuthorPrepared(
        source_row=row,
        author_name=author_name,
        author_id=_author_id_from_row(row),
        consistency_score=_consistency_score_from_row(row),
        selected_30=selected_30,
        median_views=median_views,
        threshold=threshold,
        threshold_hit_count=threshold_hit_count,
        downloaded_in_30=downloaded_in_30,
        missing_in_30=missing_in_30,
        score=score,
    )


def _choose_hits(
    selected_30: List[Dict[str, Any]],
    hits_count: int,
    threshold: float,
    strict_hits: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Returns: (hits, threshold_hits_used, fallback_hits_used)
    """
    ranked = sorted(
        selected_30,
        key=lambda v: (_to_int(v.get("playCount"), 0), _to_int(v.get("createTime"), 0)),
        reverse=True,
    )
    threshold_hits = [v for v in ranked if _to_int(v.get("playCount"), 0) >= threshold]

    if strict_hits and len(threshold_hits) < hits_count:
        return [], 0, 0

    hits = threshold_hits[:hits_count]
    threshold_hits_used = len(hits)
    if len(hits) < hits_count:
        used_ids = {str(v.get("id") or "") for v in hits}
        for v in ranked:
            vid = str(v.get("id") or "")
            if vid in used_ids:
                continue
            hits.append(v)
            used_ids.add(vid)
            if len(hits) == hits_count:
                break
    fallback_hits_used = max(0, hits_count - threshold_hits_used)
    return hits, threshold_hits_used, fallback_hits_used


def _choose_anti_hits(
    selected_30: List[Dict[str, Any]],
    hits: List[Dict[str, Any]],
    hits_count: int,
    threshold: float,
    median_views: float,
) -> List[Dict[str, Any]]:
    hit_ids = {str(v.get("id") or "") for v in hits}
    pool = [v for v in selected_30 if str(v.get("id") or "") not in hit_ids]

    normal_pool = [v for v in pool if _to_int(v.get("playCount"), 0) < threshold]
    if len(normal_pool) >= hits_count:
        pool = normal_pool

    anti_hits: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    hits_by_time = sorted(hits, key=lambda v: _to_int(v.get("createTime"), 0))
    for hit in hits_by_time:
        hit_time = _to_int(hit.get("createTime"), 0)

        candidates = [
            c for c in pool if str(c.get("id") or "") not in used_ids
        ]
        if not candidates:
            break

        best = min(
            candidates,
            key=lambda c: (
                abs(_to_int(c.get("createTime"), 0) - hit_time),
                abs(_to_int(c.get("playCount"), 0) - median_views),
                abs(_to_int(c.get("playCount"), 0)),
            ),
        )
        anti_hits.append(best)
        used_ids.add(str(best.get("id") or ""))

    return anti_hits


def build_pairs_dataset(
    input_file: Path,
    downloads_dir: Path,
    output_file: Path,
    excluded_file: Path,
    summary_file: Path,
    target_authors: int,
    videos_per_author: int,
    hits_per_author: int,
    multiplier: float,
    strict_hits: bool,
) -> None:
    rows = json.loads(input_file.read_text(encoding="utf-8-sig"))
    downloaded_index = _index_downloaded(downloads_dir)

    prepared: List[AuthorPrepared] = []
    for row in rows:
        candidate = _prepare_author(
            row=row,
            videos_per_author=videos_per_author,
            multiplier=multiplier,
            downloaded_index=downloaded_index,
        )
        if candidate is not None:
            prepared.append(candidate)

    prepared_sorted = sorted(prepared, key=lambda x: x.score, reverse=True)
    selected = prepared_sorted[:target_authors]
    excluded = prepared_sorted[target_authors:]

    selected_output: List[Dict[str, Any]] = []
    excluded_output: List[Dict[str, Any]] = []

    dropped_after_pairing = 0
    for item in selected:
        author_files = downloaded_index.get(item.author_name.lower(), {})

        hits, threshold_hits_used, fallback_hits_used = _choose_hits(
            selected_30=item.selected_30,
            hits_count=hits_per_author,
            threshold=item.threshold,
            strict_hits=strict_hits,
        )
        if len(hits) != hits_per_author:
            dropped_after_pairing += 1
            excluded_output.append(
                {
                    "author": item.author_name,
                    "author_id": item.author_id,
                    "reason": "insufficient_hits_for_strict_mode",
                    "threshold_hit_count": item.threshold_hit_count,
                }
            )
            continue

        anti_hits = _choose_anti_hits(
            selected_30=item.selected_30,
            hits=hits,
            hits_count=hits_per_author,
            threshold=item.threshold,
            median_views=item.median_views,
        )
        if len(anti_hits) != hits_per_author:
            dropped_after_pairing += 1
            excluded_output.append(
                {
                    "author": item.author_name,
                    "author_id": item.author_id,
                    "reason": "insufficient_anti_hits",
                }
            )
            continue

        hit_ids = {str(v.get("id") or "") for v in hits}
        anti_ids = {str(v.get("id") or "") for v in anti_hits}
        style = [
            v for v in item.selected_30
            if str(v.get("id") or "") not in hit_ids and str(v.get("id") or "") not in anti_ids
        ]
        if len(style) != (videos_per_author - hits_per_author * 2):
            dropped_after_pairing += 1
            excluded_output.append(
                {
                    "author": item.author_name,
                    "author_id": item.author_id,
                    "reason": "style_count_mismatch",
                    "style_count": len(style),
                }
            )
            continue

        hits_by_time = sorted(hits, key=lambda v: _to_int(v.get("createTime"), 0))

        pairs = []
        for idx, (hit, anti) in enumerate(zip(hits_by_time, anti_hits), start=1):
            hit_lite = _video_to_lite(hit, author_files.get(str(hit.get("id") or "")))
            anti_lite = _video_to_lite(anti, author_files.get(str(anti.get("id") or "")))
            pairs.append(
                {
                    "pair_index": idx,
                    "hit_video": _lite_to_dict(hit_lite),
                    "anti_hit_video": _lite_to_dict(anti_lite),
                    "time_gap_seconds": abs(hit_lite.create_time - anti_lite.create_time),
                    "hit_over_median_ratio": (
                        (hit_lite.play_count / item.median_views) if item.median_views > 0 else None
                    ),
                    "anti_over_median_ratio": (
                        (anti_lite.play_count / item.median_views) if item.median_views > 0 else None
                    ),
                }
            )

        style_lite = [
            _lite_to_dict(_video_to_lite(v, author_files.get(str(v.get("id") or ""))))
            for v in sorted(style, key=_video_sort_key, reverse=True)
        ]
        missing_ids = [
            str(v.get("id") or "")
            for v in item.selected_30
            if str(v.get("id") or "") not in author_files
        ]

        selected_output.append(
            {
                "author": {
                    "id": item.author_id,
                    "name": item.author_name,
                },
                "selection": {
                    "videos_per_author": videos_per_author,
                    "hits_per_author": hits_per_author,
                    "style_videos": len(style_lite),
                    "multiplier": multiplier,
                    "median_views": item.median_views,
                    "threshold_views": item.threshold,
                    "threshold_hit_count_in_selected_30": item.threshold_hit_count,
                    "threshold_hits_used": threshold_hits_used,
                    "fallback_hits_used": fallback_hits_used,
                },
                "download_coverage": {
                    "downloaded_in_selected_30": item.downloaded_in_30,
                    "missing_in_selected_30": item.missing_in_30,
                    "missing_video_ids": missing_ids,
                },
                "pairs": pairs,
                "style_videos": style_lite,
            }
        )

    for item in excluded:
        excluded_output.append(
            {
                "author": item.author_name,
                "author_id": item.author_id,
                "reason": f"not_in_top_{target_authors}_by_score",
                "score": {
                    "downloaded_in_selected_30": item.downloaded_in_30,
                    "threshold_hit_count": item.threshold_hit_count,
                    "consistency_score": item.consistency_score,
                },
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(
            {
                "meta": {
                    "input_file": str(input_file.as_posix()),
                    "downloads_dir": str(downloads_dir.as_posix()),
                    "target_authors": target_authors,
                    "selected_authors": len(selected_output),
                    "videos_per_author": videos_per_author,
                    "hits_per_author": hits_per_author,
                    "style_per_author": videos_per_author - hits_per_author * 2,
                    "multiplier": multiplier,
                    "strict_hits": strict_hits,
                    "dropped_after_pairing": dropped_after_pairing,
                },
                "authors": selected_output,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    excluded_file.write_text(
        json.dumps(excluded_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_pairs = len(selected_output) * hits_per_author
    total_style = len(selected_output) * (videos_per_author - hits_per_author * 2)
    missing_total = sum(
        len((row.get("download_coverage") or {}).get("missing_video_ids") or [])
        for row in selected_output
    )
    summary_lines = [
        f"Input authors: {len(rows)}",
        f"Prepared authors: {len(prepared)}",
        f"Requested selected authors: {target_authors}",
        f"Actual selected authors: {len(selected_output)}",
        f"Excluded authors: {len(excluded_output)}",
        f"Pairs total: {total_pairs}",
        f"Style videos total: {total_style}",
        f"Missing local videos across selected_30: {missing_total}",
        f"Output dataset: {output_file.as_posix()}",
        f"Excluded authors file: {excluded_file.as_posix()}",
    ]
    summary_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PVP pairs dataset: 5 hit + 5 anti-hit + 20 style videos per author."
    )
    parser.add_argument(
        "--input",
        default="datasetV/authors_shopping_pass07_full_53.json",
        help="Input JSON with author_record/videos.",
    )
    parser.add_argument(
        "--downloads-dir",
        default="datasetV/downloads_shopping",
        help="Directory with downloaded videos grouped by author.",
    )
    parser.add_argument(
        "--output",
        default="datasetV/pvp_pairs_50authors.json",
        help="Output JSON for selected authors with pairs/style split.",
    )
    parser.add_argument(
        "--excluded-output",
        default="datasetV/pvp_excluded_authors.json",
        help="Output JSON with excluded authors and reasons.",
    )
    parser.add_argument(
        "--summary-output",
        default="datasetV/pvp_pairs_50authors_summary.txt",
        help="Output summary text file.",
    )
    parser.add_argument("--target-authors", type=int, default=50)
    parser.add_argument("--videos-per-author", type=int, default=30)
    parser.add_argument("--hits-per-author", type=int, default=5)
    parser.add_argument("--multiplier", type=float, default=4.5)
    parser.add_argument(
        "--strict-hits",
        action="store_true",
        help="Require all 5 hits to satisfy playCount >= multiplier * median.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pairs_dataset(
        input_file=Path(args.input),
        downloads_dir=Path(args.downloads_dir),
        output_file=Path(args.output),
        excluded_file=Path(args.excluded_output),
        summary_file=Path(args.summary_output),
        target_authors=args.target_authors,
        videos_per_author=args.videos_per_author,
        hits_per_author=args.hits_per_author,
        multiplier=args.multiplier,
        strict_hits=args.strict_hits,
    )


if __name__ == "__main__":
    main()
