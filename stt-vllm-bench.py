#!/usr/bin/env python3
import argparse
import asyncio
import csv
import os
import sys

from src.utils import (
    wav_file_length,
    run_batch
)

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
    ap.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    ap.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    ap.add_argument("--endpoint", default="/v1/audio/transcriptions",
                    help="API endpoint path (default: /v1/audio/transcriptions)")
    ap.add_argument("-m", "--model", default="openai/whisper-large-v3-turbo", help="Model name served by vLLM")
    ap.add_argument("-f", "--filename", default="./samples/jfk.wav", help="Audio file path (wav/m4a/mp3)")
    ap.add_argument("-n", "--requests", type=int, default=50, help="Total number of requests per run (excl. warmup)")
    ap.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1, 2, 4, 8], help="Concurrency levels to test")
    ap.add_argument("--threads", type=int, default=0, help="Kept for CSV compatibility; unused for HTTP")
    ap.add_argument("--processors", type=int, default=0, help="Kept for CSV compatibility; unused for HTTP")
    ap.add_argument("--out", default="benchmark_results.csv", help="Output CSV")
    return ap.parse_args()

def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}{args.endpoint}"


    assert os.path.isfile(args.filename), f"File not found: {args.filename}"

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
                url=url,
                model=args.model,
                audio_path=args.filename,
                total_requests=args.requests,
                concurrency=conc,
            ))

            rtf = round(stats["mean_s"] / recording_len_s, 2) if stats["mean_s"] else None

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
