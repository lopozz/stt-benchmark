#!/usr/bin/env python3
import argparse
import asyncio
import csv
import os
import sys
import time
import json
import wave
import contextlib

import aiohttp

# -----------------------------
# Helpers / compatibility
# -----------------------------

first_request = True

def wav_file_length(path: str) -> float:
    with contextlib.closing(wave.open(path, "rb")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def pctl(values, p):
    if not values:
        return None
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]

# -----------------------------
# Benchmark core
# -----------------------------

async def one_request(session: aiohttp.ClientSession, url: str, model: str,
                      audio_path: str,
                      task: str = "transcribe", language: str = 'it',
                      output: str = "json"):
    global first_request
    t0 = time.perf_counter()

    params = {"task": task, "output": output, "language": language}
    data = aiohttp.FormData()
    # data.add_field("model", model)
    # data.add_field("file", open(audio_path, "rb"), filename=os.path.basename(audio_path), content_type="audio/wav")
    data.add_field("audio_file", open(audio_path, "rb"), filename=os.path.basename(audio_path), content_type="audio/wav")

    async with session.post(url, params=params, data=data,
                            headers={"accept": "application/json"}) as resp:
        text = await resp.text()
        t1 = time.perf_counter()
        ok = resp.status == 200
        if ok and first_request:
            print(json.loads(text))
            first_request = False
        return ok, text, (t1 - t0)

async def run_batch(base_url: str, model: str, audio_path: str, total_requests: int, concurrency: int):
    # url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
    url = f"{base_url.rstrip('/')}/asr"
    connector = aiohttp.TCPConnector(limit=concurrency)
    latencies = []
    errors = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm-up: measure "load time"
        _, _, _ = await one_request(session, url, model, audio_path)

        # Remaining requests
        remaining = total_requests
        sem = asyncio.Semaphore(concurrency)

        async def worker():
            nonlocal errors
            while True:
                async with sem:
                    if worker.requests_done >= remaining:
                        return
                    worker.requests_done += 1
                ok, _, elapsed = await one_request(session, url, model, audio_path)
                if ok:
                    latencies.append(elapsed)
                else:
                    errors += 1

        worker.requests_done = 0

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    total_elapsed = t_end - t_start
    completed = len(latencies)
    mean_latency_s = round(sum(latencies) / completed, 3) if completed else None
    rps = round(completed / total_elapsed, 2) if total_elapsed > 0 else 0.0

    stats = {
        "completed": completed,
        "errors": errors,
        "mean_s": mean_latency_s,
        "rps": rps,
        "total_elapsed_s": round(total_elapsed, 3),
    }
    return stats

# -----------------------------
# CLI & CSV writing (keeps your headers)
# -----------------------------

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
    ap = argparse.ArgumentParser(description="Benchmark vLLM Whisper (OpenAI API) transcription")
    ap.add_argument("--base-url", default="openai", help="Base URL of vLLM OpenAI API")
    ap.add_argument("--api", default="openai", help="API schema to use")
    ap.add_argument("-m", "--model", default="openai/whisper-large-v3-turbo", help="Model name served by vLLM")
    ap.add_argument("-f", "--filename", default="./samples/jfk.wav", help="Audio file path (wav/m4a/mp3)")
    ap.add_argument("-f", "--filename", default="./samples/jfk.wav", help="Audio file path (wav/m4a/mp3)")
    ap.add_argument("-n", "--requests", type=int, default=50, help="Total number of requests per run (excl. warmup)")
    ap.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1, 2, 4, 8], help="Concurrency levels to test")
    ap.add_argument("--threads", type=int, default=0, help="Kept for CSV compatibility; unused for HTTP")
    ap.add_argument("--processors", type=int, default=0, help="Kept for CSV compatibility; unused for HTTP")
    ap.add_argument("--out", default="benchmark_results.csv", help="Output CSV")
    return ap.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.filename):
        print(f"Audio file not found: {args.filename}", file=sys.stderr)
        sys.exit(1)

    recording_len_s = wav_file_length(args.filename)

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if new_file:
            writer.writeheader()

        # Run each concurrency level
        for conc in args.concurrency:
            print(f"Benchmarking conc={conc} n={args.requests} file={args.filename} recording_len_s={recording_len_s:.2f}s...")
            stats = asyncio.run(run_batch(
                base_url=args.base_url,
                model=args.model,
                audio_path=args.filename,
                total_requests=args.requests,
                concurrency=conc,
            ))

            rtf = round(stats["mean_s"] / recording_len_s, 2) if stats["mean_s"] else None

            # Map to your original columns
            row = {
                "Model": args.model,
                "Recording Length (s)": round(recording_len_s, 3),
                "Concurrency": conc,
                "Requests": args.requests,
                "Mean Latency (s)": stats["mean_s"],
                "RPS": stats["rps"],
                "RTF": rtf,
                "Errors": stats["errors"],
            }
            writer.writerow(row)

            print(f"  mean={stats['mean_s']}s  RPS={stats['rps']}  RTF={rtf}  errors={stats['errors']}")

if __name__ == "__main__":
    main()
