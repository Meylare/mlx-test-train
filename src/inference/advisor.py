"""
Timecode Advisor (Block D) — MLX port.

Analyses 3-D ``delta_z`` from the optimiser and generates human-readable
advice tied to specific moments in the video.
"""

import mlx.core as mx
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ChangeIntensity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SegmentAnalysis:
    segment_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    delta_norm: float
    max_delta_norm: float
    intensity: ChangeIntensity
    direction_vector: mx.array
    contribution_percent: float


@dataclass
class FrameAdvice:
    frame_idx: int
    time_sec: float
    delta_norm: float
    rank: int
    advice_text: str


@dataclass
class VideoAdvice:
    video_duration_sec: float
    total_frames: int
    overall_score_improvement: float
    segments: List[SegmentAnalysis]
    top_frames: List[FrameAdvice]
    summary: str
    detailed_advice: List[str]


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

class TimecodeAdvisor:
    """Analyses ``delta_z`` and produces timecoded advice."""

    def __init__(
        self,
        fps: float = 2.0,
        num_segments: int = 3,
        intensity_thresholds: Optional[Dict[str, float]] = None,
        top_k_frames: int = 5,
    ):
        self.fps = fps
        self.num_segments = num_segments
        self.top_k_frames = top_k_frames
        self.intensity_thresholds = intensity_thresholds or {
            "critical": 0.8,
            "high": 0.5,
            "medium": 0.2,
            "low": 0.05,
        }

        self.segment_templates = {
            0: {
                ChangeIntensity.CRITICAL: "В начале видео (первые {duration}с) требуются КРИТИЧЕСКИЕ изменения! Хук не работает - полностью пересмотри первые кадры.",
                ChangeIntensity.HIGH: "Начало видео ({start}-{end}с) нуждается в серьезной доработке. Усиль визуальный хук, добавь динамики.",
                ChangeIntensity.MEDIUM: "В начале ({start}-{end}с) можно улучшить привлечение внимания. Попробуй более яркие цвета или неожиданный ракурс.",
                ChangeIntensity.LOW: "Начало видео неплохое, но можно немного усилить.",
                ChangeIntensity.NONE: "Начало видео отличное, оставь как есть!",
            },
            1: {
                ChangeIntensity.CRITICAL: "Середина видео ({start}-{end}с) теряет зрителя! Нужна полная переработка этого сегмента.",
                ChangeIntensity.HIGH: "В середине ({start}-{end}с) падает вовлеченность. Добавь поворот сюжета или смену темпа.",
                ChangeIntensity.MEDIUM: "Середина видео ({start}-{end}с) может быть динамичнее. Сократи паузы, добавь визуальных переходов.",
                ChangeIntensity.LOW: "Середина хорошая, возможны минимальные улучшения.",
                ChangeIntensity.NONE: "Середина видео работает отлично!",
            },
            2: {
                ChangeIntensity.CRITICAL: "Концовка ({start}-{end}с) провальная! Нужен сильный CTA или неожиданный финал.",
                ChangeIntensity.HIGH: "Конец видео ({start}-{end}с) слабый. Добавь запоминающийся момент или призыв к действию.",
                ChangeIntensity.MEDIUM: "Концовку ({start}-{end}с) можно сделать ярче. Усиль финальный акцент.",
                ChangeIntensity.LOW: "Концовка неплохая, можно немного усилить финал.",
                ChangeIntensity.NONE: "Отличная концовка!",
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        delta_z: mx.array,
        per_frame_delta_norm: Optional[mx.array] = None,
        score_improvement: float = 0.0,
        video_duration_sec: Optional[float] = None,
    ) -> VideoAdvice:
        # Remove batch dim if present
        if delta_z.ndim == 3:
            delta_z = delta_z[0]
        if per_frame_delta_norm is not None and per_frame_delta_norm.ndim == 2:
            per_frame_delta_norm = per_frame_delta_norm[0]

        num_frames = delta_z.shape[0]

        if per_frame_delta_norm is None:
            per_frame_delta_norm = mx.sqrt(
                mx.sum(delta_z * delta_z, axis=-1) + 1e-8
            )

        if video_duration_sec is None:
            video_duration_sec = num_frames / self.fps

        segments = self._analyze_segments(
            delta_z, per_frame_delta_norm, num_frames, video_duration_sec,
        )
        top_frames = self._get_top_frames(
            per_frame_delta_norm, num_frames, video_duration_sec,
        )
        detailed_advice = self._generate_segment_advice(segments)
        summary = self._generate_summary(segments, score_improvement)

        return VideoAdvice(
            video_duration_sec=video_duration_sec,
            total_frames=num_frames,
            overall_score_improvement=score_improvement,
            segments=segments,
            top_frames=top_frames,
            summary=summary,
            detailed_advice=detailed_advice,
        )

    # ------------------------------------------------------------------
    # Segment analysis
    # ------------------------------------------------------------------

    def _analyze_segments(
        self,
        delta_z: mx.array,
        per_frame_delta_norm: mx.array,
        num_frames: int,
        video_duration_sec: float,
    ) -> List[SegmentAnalysis]:
        segments = []
        frames_per_seg = num_frames // self.num_segments

        max_norm = float(mx.max(per_frame_delta_norm).item())
        if max_norm < 1e-8:
            max_norm = 1.0
        total_delta_norm = float(mx.sum(per_frame_delta_norm).item())

        for seg_idx in range(self.num_segments):
            sf = seg_idx * frames_per_seg
            ef = (seg_idx + 1) * frames_per_seg if seg_idx < self.num_segments - 1 else num_frames

            start_time = sf / self.fps
            end_time = ef / self.fps

            seg_norms = per_frame_delta_norm[sf:ef]
            seg_delta = delta_z[sf:ef]

            delta_norm_val = float(mx.mean(seg_norms).item())
            max_delta_val = float(mx.max(seg_norms).item())

            seg_total = float(mx.sum(seg_norms).item())
            contribution = (seg_total / total_delta_norm * 100) if total_delta_norm > 1e-8 else 0.0

            # Direction vector (normalised mean)
            direction = mx.mean(seg_delta, axis=0)
            dir_norm = mx.sqrt(mx.sum(direction * direction) + 1e-8)
            direction = direction / dir_norm

            intensity = self._classify_intensity(delta_norm_val / max_norm)

            segments.append(SegmentAnalysis(
                segment_id=seg_idx,
                start_frame=sf,
                end_frame=ef,
                start_time_sec=start_time,
                end_time_sec=end_time,
                delta_norm=delta_norm_val,
                max_delta_norm=max_delta_val,
                intensity=intensity,
                direction_vector=direction,
                contribution_percent=contribution,
            ))
        return segments

    def _classify_intensity(self, normalised: float) -> ChangeIntensity:
        if normalised >= self.intensity_thresholds["critical"]:
            return ChangeIntensity.CRITICAL
        if normalised >= self.intensity_thresholds["high"]:
            return ChangeIntensity.HIGH
        if normalised >= self.intensity_thresholds["medium"]:
            return ChangeIntensity.MEDIUM
        if normalised >= self.intensity_thresholds["low"]:
            return ChangeIntensity.LOW
        return ChangeIntensity.NONE

    # ------------------------------------------------------------------
    # Top frames
    # ------------------------------------------------------------------

    def _get_top_frames(
        self,
        per_frame_delta_norm: mx.array,
        num_frames: int,
        video_duration_sec: float,
    ) -> List[FrameAdvice]:
        k = min(self.top_k_frames, num_frames)

        # MLX doesn't have topk — use argsort descending
        sorted_indices = mx.argsort(-per_frame_delta_norm)
        top_indices = sorted_indices[:k]
        top_values = per_frame_delta_norm[top_indices]

        top_frames = []
        for rank, (idx_arr, norm_arr) in enumerate(
            zip(top_indices.tolist(), top_values.tolist()), 1
        ):
            idx = int(idx_arr) if not isinstance(idx_arr, int) else idx_arr
            norm_val = float(norm_arr) if not isinstance(norm_arr, float) else norm_arr
            time_sec = idx / self.fps
            advice_text = self._generate_frame_advice(idx, time_sec, norm_val, rank, num_frames)
            top_frames.append(FrameAdvice(
                frame_idx=idx,
                time_sec=time_sec,
                delta_norm=norm_val,
                rank=rank,
                advice_text=advice_text,
            ))
        return top_frames

    def _generate_frame_advice(
        self, frame_idx: int, time_sec: float, delta_norm: float,
        rank: int, total_frames: int,
    ) -> str:
        ratio = frame_idx / max(total_frames, 1)
        if ratio < 0.2:
            pos, hint = "в самом начале", "Усиль хук"
        elif ratio < 0.4:
            pos, hint = "в начале", "Добавь динамики"
        elif ratio < 0.6:
            pos, hint = "в середине", "Удерживай внимание"
        elif ratio < 0.8:
            pos, hint = "ближе к концу", "Готовь к финалу"
        else:
            pos, hint = "в конце", "Усиль концовку"
        return f"[{time_sec:.1f}с] {pos}: {hint}. Сила изменения: {delta_norm:.2f}"

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def _generate_segment_advice(self, segments: List[SegmentAnalysis]) -> List[str]:
        result = []
        for seg in segments:
            tid = min(seg.segment_id, 2)
            templates = self.segment_templates.get(tid, self.segment_templates[2])
            tmpl = templates.get(seg.intensity, "")
            if tmpl:
                text = tmpl.format(
                    start=f"{seg.start_time_sec:.1f}",
                    end=f"{seg.end_time_sec:.1f}",
                    duration=f"{seg.end_time_sec - seg.start_time_sec:.1f}",
                )
                if seg.contribution_percent > 40:
                    text += f" (вклад в общую дельту: {seg.contribution_percent:.0f}% - основной фокус!)"
                result.append(text)
        return result

    def _generate_summary(
        self, segments: List[SegmentAnalysis], score_improvement: float,
    ) -> str:
        worst = max(segments, key=lambda s: s.delta_norm)
        critical_count = sum(
            1 for s in segments
            if s.intensity in (ChangeIntensity.CRITICAL, ChangeIntensity.HIGH)
        )
        if score_improvement > 0.5:
            imp = f"Потенциальное улучшение скора: +{score_improvement:.2f} (значительное!)"
        elif score_improvement > 0.1:
            imp = f"Потенциальное улучшение скора: +{score_improvement:.2f} (хорошее)"
        else:
            imp = f"Потенциальное улучшение скора: +{score_improvement:.2f} (небольшое)"

        names = {0: "начало", 1: "середина", 2: "конец"}
        if critical_count == 0:
            status = "Видео в хорошем состоянии, минимальные правки."
        elif critical_count == 1:
            seg_name = names.get(worst.segment_id, f"сегмент {worst.segment_id}")
            status = (f"Основной фокус: {seg_name} "
                      f"({worst.start_time_sec:.1f}-{worst.end_time_sec:.1f}с).")
        else:
            status = f"Требуется переработка {critical_count} из {len(segments)} сегментов."

        return f"{imp} {status}"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_advice(self, advice: VideoAdvice, verbose: bool = True) -> str:
        lines = [
            "=" * 60,
            "АНАЛИЗ ВИДЕО - СОВЕТЫ ПО ТАЙМКОДАМ",
            "=" * 60,
            "",
            f"Длительность: {advice.video_duration_sec:.1f}с ({advice.total_frames} кадров)",
            "",
            "РЕЗЮМЕ:",
            advice.summary,
            "",
        ]
        if advice.detailed_advice:
            lines.append("ДЕТАЛЬНЫЕ СОВЕТЫ ПО СЕГМЕНТАМ:")
            for i, adv in enumerate(advice.detailed_advice, 1):
                lines.append(f"  {i}. {adv}")
            lines.append("")
        if verbose and advice.top_frames:
            lines.append("ТОП МОМЕНТОВ ДЛЯ ИЗМЕНЕНИЙ:")
            for frame in advice.top_frames:
                lines.append(f"  #{frame.rank}: {frame.advice_text}")
            lines.append("")
        if verbose:
            lines.append("СТАТИСТИКА ПО СЕГМЕНТАМ:")
            for seg in advice.segments:
                lines.append(
                    f"  [{seg.start_time_sec:.1f}-{seg.end_time_sec:.1f}с] "
                    f"Интенсивность: {seg.intensity.value.upper()}, "
                    f"Норма: {seg.delta_norm:.3f}, "
                    f"Вклад: {seg.contribution_percent:.1f}%"
                )
        lines.append("=" * 60)
        return "\n".join(lines)
