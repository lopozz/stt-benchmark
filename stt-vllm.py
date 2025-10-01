#!/usr/bin/env python3
import json
import asyncio
import aiohttp
import argparse
from pathlib import Path
import tempfile

import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

DEFAULT_URL = "http://localhost:8000/v1/audio/transcriptions"
MAX_LEN_S = 29.5  # keep under server's 30s limit

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch transcription to an OpenAI-compatible /v1/audio/transcriptions endpoint (with simple Silero VAD fallback)."
    )
    p.add_argument("--url", default=DEFAULT_URL, help=f"Transcriptions endpoint (default: {DEFAULT_URL})")
    p.add_argument("--indir", required=True, help="Directory with audio files.")
    p.add_argument("--outdir", required=True, help="Directory to save .txt transcripts.")
    p.add_argument("--model", default="openai/whisper-large-v3-turbo", help="Model served by vLLM.")
    p.add_argument("--language", default=None, help="Optional language code (e.g., it, en).")
    p.add_argument("--extensions", nargs="*", default=[".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"],
                   help="Audio file extensions to include.")
    return p.parse_args()

def find_audio_files(indir: str, exts):
    base = Path(indir)
    if not base.is_dir():
        raise SystemExit(f"Input dir not found: {indir}")
    exts = {e.lower() for e in exts}
    files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise SystemExit(f"No audio files found in {indir} matching {sorted(exts)}")
    return files

def group_speech_segments(timestamps, max_len=MAX_LEN_S):
    """Concat consecutive speech segments until adding one would exceed max_len."""
    if not timestamps:
        return []
    chunks = []
    cur_start = timestamps[0]["start"]
    cur_end = timestamps[0]["end"]
    for seg in timestamps[1:]:
        # if the next segment would push the chunk over max_len, flush current
        if (seg["end"] - cur_start) > max_len:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = seg["start"], seg["end"]
        else:
            cur_end = seg["end"]
    chunks.append((cur_start, cur_end))
    return chunks

async def post_one(session: aiohttp.ClientSession, url: str, model: str,
                   language: str | None, file_path: Path) -> str:
    form = aiohttp.FormData()
    form.add_field("file", file_path.open("rb"),
                   filename=file_path.name,
                   content_type="application/octet-stream")
    form.add_field("model", model)
    form.add_field("response_format", "json")
    if language:
        form.add_field("language", language)
    async with session.post(url, data=form) as resp:
        body = await resp.text()
        if resp.status != 200:
            # bubble up status + body for the caller to decide
            raise RuntimeError(f"HTTP {resp.status}::{body}")
        try:
            return json.loads(body).get("text", body).strip()
        except json.JSONDecodeError:
            return body.strip()

def write_wav_slice(src: Path, start_s: float, end_s: float, dst: Path):
    """Slice [start_s, end_s] seconds to WAV using soundfile (keeps original sample rate)."""
    audio, sr = sf.read(str(src), always_2d=False)
    if audio.ndim > 1:
        # simple mono mixdown
        import numpy as np
        audio = audio.mean(axis=1).astype(audio.dtype)
    i0 = max(0, int(start_s * sr))
    i1 = min(len(audio), int(end_s * sr))
    clip = audio[i0:i1]
    sf.write(str(dst), clip, sr)

async def transcribe_file(session: aiohttp.ClientSession, url: str, model: str,
                          language: str | None, src: Path, dst: Path):
    # 1) Try whole file
    try:
        text = await post_one(session, url, model, language, src)
        dst.write_text(text, encoding="utf-8")
        print(f"[ok] {src.name} -> {dst.name}")
        return
    except RuntimeError as e:
        err = str(e)

    # 2) If clip too long, do simple VAD-based chunking
    if "Maximum clip duration" in err or "30s" in err or "HTTP 400" in err:
        vad_model = load_silero_vad()
        wav = read_audio(str(src))  # 16k mono tensor via torchaudio backend
        ts = get_speech_timestamps(wav, vad_model, return_seconds=True)  # [{'start': s, 'end': e}, ...]

        if not ts:
            # No speech detected; just rethrow the original error
            raise RuntimeError(f"{src.name}: no speech detected and original error: {err}")

        ranges = group_speech_segments(ts, max_len=MAX_LEN_S)

        parts = []
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            for i, (s, e) in enumerate(ranges, 1):
                chunk_path = tmpdir / f"chunk_{i:05d}.wav"
                write_wav_slice(src, s, e, chunk_path)
                try:
                    part = await post_one(session, url, model, language, chunk_path)
                    parts.append(part)
                    print(f"  [chunk {i}/{len(ranges)} ok] {chunk_path.name} ({e - s:.2f}s)")
                except Exception as ce:
                    raise RuntimeError(f"{src.name}: chunk {i} failed: {ce}") from ce

        full = " ".join(p.strip() for p in parts if p.strip())
        dst.write_text(full, encoding="utf-8")
        print(f"[ok-vad] {src.name} -> {dst.name} ({len(ranges)} chunks)")
        return

    # Other errors
    raise RuntimeError(f"{src.name}: {err}")

async def main_async(args):
    files = find_audio_files(args.indir, args.extensions)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=0)  # let server handle concurrency
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for src in files:
            dst = outdir / (src.with_suffix(".txt").name)
            tasks.append(transcribe_file(session, args.url, args.model, args.language, src, dst))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = sum(1 for r in results if not isinstance(r, Exception))
    errs = [e for e in results if isinstance(e, Exception)]
    if errs:
        print("\nErrors:")
        for e in errs:
            print("-", e)
    print(f"\nDone. OK={ok} ERR={len(errs)} Total={len(results)}")

def main():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
