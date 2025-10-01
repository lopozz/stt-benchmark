import os
import time
import json
import wave
import asyncio
import aiohttp
import contextlib




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
    data.add_field("model", model)
    data.add_field("file", open(audio_path, "rb"), filename=os.path.basename(audio_path), content_type="audio/wav")
    # data.add_field("audio_file", open(audio_path, "rb"), filename=os.path.basename(audio_path), content_type="audio/wav")

    async with session.post(url, params=params, data=data,
                            headers={"accept": "application/json"}) as resp:
        text = await resp.text()
        t1 = time.perf_counter()
        ok = resp.status == 200
        if ok and first_request:
            first_request = False
        return ok, text, (t1 - t0)

async def run_batch(url: str, model: str, audio_path: str, total_requests: int, concurrency: int):
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