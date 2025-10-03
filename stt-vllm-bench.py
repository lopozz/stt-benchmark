
#!/usr/bin/env python3
import argparse
import asyncio
import csv
import mimetypes
import os
import sys
import time
import wave
import contextlib

import aiohttp

CSV_HEADERS = [
    "Model",
    "Recording Length (s)",
    "Requests",
    "Concurrency",
    "Mean Latency (s)",
    "RPS",
    "RTF",
    "Errors",
]

def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark transcription endpoint with per-user sequential requests")
    ap.add_argument("--url", required=True, help="e.g. http://localhost:8000/v1/audio/transcriptions")
    ap.add_argument("-H", "--header", action="append", default=[], help="Custom header(s) 'Key:Value'. Can be used multiple times.")
    ap.add_argument("-m", "--model", default="openai/whisper-large-v3-turbo", help="Model name")
    ap.add_argument("-f", "--filename", default="./samples/jfk.wav", help="Audio file path (wav/m4a/mp3)")
    ap.add_argument("-n", "--requests", type=int, default=3, help="Sequential requests per user (per worker)")
    ap.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1, 2, 3], help="Concurrency levels to test")
    ap.add_argument("--threads", type=int, default=0, help="(unused, kept for CSV compatibility)")
    ap.add_argument("--processors", type=int, default=0, help="(unused, kept for CSV compatibility)")
    ap.add_argument("--out", default="benchmark_results.csv", help="Output CSV path")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Whisper task")
    ap.add_argument("--language", default="it", help="Source language code")
    ap.add_argument("--output", default="json", help="Server response format hint, e.g. json or srt")
    return ap.parse_args()

def wav_file_length_seconds(path: str) -> float | None:
    # Only accurate for WAV. For non-WAV, return None so RTF is omitted.
    if not path.lower().endswith(".wav"):
        return None
    with contextlib.closing(wave.open(path, "rb")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

async def one_request(session: aiohttp.ClientSession, url: str, model: str,
                      audio_path: str, headers: dict | None,
                      task: str, language: str, output: str) -> tuple[bool, float]:
    t0 = time.perf_counter()

    request_headers = {"accept": "application/json"}
    if headers:
        request_headers.update(headers)

    content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"

    data = aiohttp.FormData()
    data.add_field("model", model)
    data.add_field("task", task)
    data.add_field("language", language)
    data.add_field("output", output)
    # keep the file open until the request finishes
    with open(audio_path, "rb") as fh:
        data.add_field("file", fh, filename=os.path.basename(audio_path), content_type=content_type)
        async with session.post(url, data=data, headers=request_headers) as resp:
            await resp.read()  # we only measure latency; discard body
            elapsed = time.perf_counter() - t0
            return (resp.status == 200), elapsed

async def run_batch(url: str, model: str, audio_path: str, requests_per_user: int,
                    concurrency: int, headers: dict | None,
                    task: str, language: str, output: str):
    connector = aiohttp.TCPConnector(limit=concurrency)
    latencies: list[float] = []
    errors = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker(user_id: int):
            nonlocal errors
            for _ in range(requests_per_user):
                ok, elapsed = await one_request(session, url, model, audio_path, headers, task, language, output)
                if ok:
                    latencies.append(elapsed)
                else:
                    errors += 1

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    total_elapsed = t_end - t_start
    completed = len(latencies)
    mean_latency_s = round(sum(latencies) / completed, 4) if completed else None
    rps = round(completed / total_elapsed, 3) if total_elapsed > 0 else 0.0

    return {
        "completed": completed,
        "errors": errors,
        "mean_s": mean_latency_s,
        "rps": rps,
        "total_elapsed_s": round(total_elapsed, 3),
    }

def main():
    args = parse_args()
    assert os.path.isfile(args.filename), f"File not found: {args.filename}"

    # Parse custom headers
    headers = {}
    for h in args.header or []:
        if ':' not in h:
            print(f"Warning: '{h}' is not in 'Key:Value' format. Skipping.", file=sys.stderr)
            continue
        k, v = h.split(':', 1)
        headers[k.strip()] = v.strip()

    # Recording length (used for RTF if WAV)
    recording_len_s = wav_file_length_seconds(args.filename)

    # Prepare CSV
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if new_file:
            writer.writeheader()

        for conc in args.concurrency:
            total_requests = args.requests * conc
            print(f"Running: conc={conc} users, {args.requests} requests each (total={total_requests}) | file={args.filename}")

            stats = asyncio.run(
                run_batch(
                    url=args.url,
                    model=args.model,
                    audio_path=args.filename,
                    requests_per_user=args.requests,
                    concurrency=conc,
                    headers=headers,
                    task=args.task,
                    language=args.language,
                    output=args.output,
                )
            )

            rtf = round(stats["mean_s"] / recording_len_s, 3) if (stats["mean_s"] and recording_len_s) else None

            row = {
                "Model": args.model,
                "Recording Length (s)": round(recording_len_s, 3) if recording_len_s else "",
                "Concurrency": conc,
                "Requests": total_requests,
                "Mean Latency (s)": stats["mean_s"],
                "RPS": stats["rps"],
                "RTF": rtf,
                "Errors": stats["errors"],
            }
            writer.writerow(row)

            print(f"  completed={stats['completed']}/{total_requests}  mean={stats['mean_s']}s  RPS={stats['rps']}  RTF={rtf}  errors={stats['errors']}")

if __name__ == "__main__":
    main()