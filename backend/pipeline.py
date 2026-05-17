"""The real Phase 2 pipeline: YouTube URL -> structured summary JSON.

Three stages, each a single function:
  1. download_audio(url, out_dir)  -> (wav_path, metadata)
  2. transcribe(wav_path)          -> {full_text, word_count, language}
  3. summarize(transcript, title)  -> (SummaryContent, usage_info)

run_pipeline(url) orchestrates all three and returns a dict shaped like
SummaryPayload (see models.py).

External dependencies:
  - yt_dlp (Python lib)             https://github.com/yt-dlp/yt-dlp
  - whisper.cpp CLI                 /opt/homebrew/bin/whisper
  - ffmpeg (used by yt-dlp)         auto-detected via PATH
  - anthropic SDK                   https://github.com/anthropics/anthropic-sdk-python
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import anthropic
import yt_dlp

from models import (
    SummaryContent,
    SummaryPayload,
    TranscriptInfo,
    VideoMetadata,
    ResponseMetadata,
)

logger = logging.getLogger(__name__)

# ---------- configuration ----------

# whisper binary: check PATH first, then fallback to common locations
def _find_whisper_bin() -> str:
    env_val = os.getenv("WHISPER_BIN", "")
    if env_val:
        return env_val
    for candidate in ["whisper", "/opt/homebrew/bin/whisper", "/usr/local/bin/whisper"]:
        if shutil.which(candidate) or Path(candidate).exists():
            return candidate
    return "whisper"  # will fail loudly at transcribe time

WHISPER_BIN = _find_whisper_bin()
WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    str(Path.home() / ".cache/hyperframes/whisper/models/ggml-small.en.bin"),
)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_MAX_TOKENS = 1500

SYSTEM_PROMPT = (
    "You are a precise video summarizer. Given a transcript, you extract "
    "structured insights for busy professionals. You return STRICT JSON "
    "matching the schema. Insights must be grounded in the transcript "
    "(do not invent facts). When the transcript is ambiguous, say so."
)

USER_INSTRUCTION_TEMPLATE = """\
The transcript above is from a YouTube video titled: "{title}".

Return ONLY a JSON object with this exact shape (no markdown, no prose around it):

{{
  "executive_summary": "2-3 paragraph summary, ~150 words, for a busy professional",
  "key_insights": ["5 most important takeaways, each 1 sentence"],
  "action_items": ["concrete next steps the viewer should take, if any"],
  "topics_covered": ["3-7 short topic tags, e.g. 'machine learning', 'pricing strategy'"],
  "tone": "academic | casual | technical | promotional | tutorial | interview | other"
}}

Rules:
- key_insights MUST have exactly 5 items unless the transcript is too short
- action_items can be empty [] if the video doesn't suggest actions
- topics_covered MUST have at least 3 items
- All claims must be grounded in the transcript
"""


# ---------- 1. yt-dlp wrapper ----------

class DownloadError(Exception):
    """Raised when yt-dlp can't extract the video."""


def download_audio(url: str, out_dir: Path) -> Tuple[Path, dict]:
    """Download the audio track of a YouTube video as a WAV file.

    Returns (wav_path, metadata) where metadata is yt-dlp's info dict
    (title, channel/uploader, duration, thumbnail, language).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Try multiple player clients — iOS/Android bypass most web-tier 403s
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "android", "web"],
            }
        },
        # Add a browser-like User-Agent to reduce bot detection
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            )
        },
        # Retry on transient network failures
        "retries": 3,
        "fragment_retries": 3,
    }

    logger.info("yt-dlp: downloading audio from %s", url)
    t0 = time.time()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        err_str = str(e).lower()
        logger.error("yt-dlp DownloadError: %s", e)
        if "private" in err_str:
            raise DownloadError("This video is private and cannot be accessed.") from e
        if "age" in err_str:
            raise DownloadError("This video is age-restricted and cannot be downloaded.") from e
        if "unavailable" in err_str or "not available" in err_str:
            raise DownloadError("This video is unavailable in your region or has been removed.") from e
        raise DownloadError(f"Couldn't download this video: {e}") from e
    except Exception as e:
        logger.exception("yt-dlp unexpected error")
        raise DownloadError(f"Unexpected error during download: {e}") from e

    elapsed = time.time() - t0
    logger.info("yt-dlp: completed in %.1fs", elapsed)

    wav_path = out_dir / "audio.wav"
    if not wav_path.exists():
        # Fallback: glob for any audio file written
        candidates = list(out_dir.glob("audio.*"))
        if not candidates:
            raise DownloadError("yt-dlp succeeded but no audio file was written.")
        wav_path = candidates[0]
        logger.warning("Expected audio.wav but found %s", wav_path.name)

    return wav_path, info or {}


# ---------- 2. whisper wrapper ----------

class TranscribeError(Exception):
    """Raised when whisper fails."""


def transcribe(audio_path: Path) -> dict:
    """Run whisper.cpp on the audio file and return transcript info.

    Returns: {"full_text": str, "word_count": int, "language": "en"}
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise TranscribeError(f"Audio file not found: {audio_path}")

    whisper_model = Path(WHISPER_MODEL)
    if not whisper_model.exists():
        raise TranscribeError(
            f"Whisper model not found at {WHISPER_MODEL}. "
            f"Set WHISPER_MODEL env var or install ggml-small.en.bin."
        )
    if not shutil.which(WHISPER_BIN) and not Path(WHISPER_BIN).exists():
        raise TranscribeError(
            f"Whisper binary not found at '{WHISPER_BIN}'. "
            f"Install whisper.cpp or set WHISPER_BIN env var."
        )

    logger.info("whisper: transcribing %s (%.1f MB)", audio_path.name,
                audio_path.stat().st_size / 1_048_576)
    t0 = time.time()
    cmd = [
        WHISPER_BIN,
        "-m", str(whisper_model),
        "-f", str(audio_path),
        "-oj",       # output JSON beside the audio file
        "-l", "en",  # force English
        "-nt",       # no timestamps in text
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        raise TranscribeError("Whisper transcription timed out after 10 minutes.") from e
    if proc.returncode != 0:
        logger.error("whisper stderr: %s", proc.stderr[-500:])
        raise TranscribeError(f"Whisper failed (exit {proc.returncode}): {proc.stderr[-200:]}")

    elapsed = time.time() - t0
    logger.info("whisper: completed in %.1fs", elapsed)

    # whisper.cpp writes <audio>.json beside the input file
    json_path = audio_path.with_suffix(audio_path.suffix + ".json")
    if not json_path.exists():
        json_path = audio_path.parent / (audio_path.stem + ".json")
    if not json_path.exists():
        raise TranscribeError(f"Whisper finished but no JSON output found near {audio_path}.")

    data = json.loads(json_path.read_text())
    # whisper.cpp output: {"transcription": [{"text": "...", "offsets": {...}}, ...]}
    segments = data.get("transcription") or data.get("segments") or []
    full_text = " ".join(s.get("text", "").strip() for s in segments).strip()
    full_text = re.sub(r"\s+", " ", full_text)

    if not full_text:
        raise TranscribeError("Whisper returned an empty transcript — video may be silent.")

    return {
        "full_text": full_text,
        "word_count": len(full_text.split()),
        "language": "en",
    }


# ---------- 3. Claude summarizer with prompt caching ----------

class SummarizeError(Exception):
    """Raised when Claude returns an unusable response."""


def summarize(
    transcript_text: str,
    video_title: str,
    client: anthropic.Anthropic,
) -> Tuple[SummaryContent, dict]:
    """Send transcript to Claude with prompt caching, return parsed summary + usage.

    The transcript content block is marked with cache_control: ephemeral so a
    second call (same transcript, different instructions) is ~90% cheaper.
    """
    if not transcript_text or not transcript_text.strip():
        raise SummarizeError("Transcript is empty — nothing to summarize.")

    user_instruction = USER_INSTRUCTION_TEMPLATE.format(title=video_title or "Untitled video")

    logger.info(
        "Claude: model=%s transcript_chars=%d",
        ANTHROPIC_MODEL, len(transcript_text),
    )
    t0 = time.time()
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<transcript>\n{transcript_text}\n</transcript>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": user_instruction},
                    ],
                }
            ],
        )
    except anthropic.AuthenticationError as e:
        logger.error("Anthropic AuthenticationError")
        raise SummarizeError("Invalid ANTHROPIC_API_KEY. Check your backend/.env.") from e
    except anthropic.BadRequestError as e:
        msg = str(e).lower()
        if "credit balance" in msg:
            raise SummarizeError(
                "Anthropic API credits exhausted. Add credits at "
                "https://console.anthropic.com/settings/billing"
            ) from e
        raise SummarizeError(f"Anthropic bad request: {e}") from e
    except anthropic.APIError as e:
        logger.error("Anthropic APIError: %s", e)
        raise SummarizeError(f"AI service error: {e}") from e

    elapsed = time.time() - t0
    logger.info("Claude: completed in %.1fs", elapsed)

    if not response.content or not response.content[0].text:
        raise SummarizeError("Empty response from Claude.")

    raw = response.content[0].text.strip()
    # Claude sometimes wraps JSON in prose — extract just the JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.error("Claude response had no JSON. raw[:300]=%s", raw[:300])
        raise SummarizeError("AI returned an unexpected response format.")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error("Claude JSON parse failed: %s | raw=%s", e, raw[:300])
        raise SummarizeError("AI returned malformed JSON.") from e

    try:
        summary = SummaryContent(
            executive_summary=parsed.get("executive_summary", ""),
            key_insights=parsed.get("key_insights", []) or [],
            action_items=parsed.get("action_items", []) or [],
            topics_covered=parsed.get("topics_covered", []) or [],
            tone=parsed.get("tone"),
        )
    except Exception as e:
        raise SummarizeError(f"AI response missing required fields: {e}") from e

    usage = getattr(response, "usage", None)
    cache_read   = getattr(usage, "cache_read_input_tokens",   0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    input_tok    = getattr(usage, "input_tokens",  0) or 0
    output_tok   = getattr(usage, "output_tokens", 0) or 0
    total = input_tok + cache_create + cache_read + output_tok

    return summary, {
        "tokens_used":            total,
        "cache_hit":              cache_read > 0,
        "input_tokens":           input_tok,
        "cache_creation_tokens":  cache_create,
        "cache_read_tokens":      cache_read,
        "output_tokens":          output_tok,
    }


# ---------- orchestration ----------

def run_pipeline(url: str, out_dir: Path, client: anthropic.Anthropic) -> dict:
    """End-to-end: URL -> SummaryPayload dict.

    Caller is responsible for cleaning up `out_dir` (use tempfile.TemporaryDirectory).
    """
    pipeline_start = time.time()

    # 1. Download
    wav_path, info = download_audio(url, out_dir)

    # 2. Transcribe
    transcript = transcribe(wav_path)

    # 3. Summarize
    title = info.get("title") or "Untitled video"
    summary_content, usage = summarize(transcript["full_text"], title, client)

    pipeline_seconds = time.time() - pipeline_start

    payload = SummaryPayload(
        video=VideoMetadata(
            title=title,
            channel=info.get("channel") or info.get("uploader") or "Unknown",
            duration_seconds=int(info.get("duration") or 0),
            url=url,
            thumbnail_url=info.get("thumbnail"),
        ),
        transcript=TranscriptInfo(
            full_text=transcript["full_text"],
            word_count=transcript["word_count"],
            language=transcript["language"],
        ),
        summary=summary_content,
        metadata=ResponseMetadata(
            generated_at=datetime.now(timezone.utc).isoformat(),
            model=ANTHROPIC_MODEL,
            tokens_used=usage["tokens_used"],
            cache_hit=usage["cache_hit"],
            input_tokens=usage["input_tokens"],
            cache_creation_tokens=usage["cache_creation_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            output_tokens=usage["output_tokens"],
            pipeline_seconds=round(pipeline_seconds, 2),
        ),
    )

    logger.info(
        "pipeline: ok | seconds=%.1f | tokens=%d | cache_hit=%s",
        pipeline_seconds, usage["tokens_used"], usage["cache_hit"],
    )
    return payload.model_dump()
