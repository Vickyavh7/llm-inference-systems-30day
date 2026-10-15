import asyncio
import time

async def send_simulated_request(request_id):
    t0 = time.perf_counter()
    # In live setup: replace with httpx call to http://localhost:8000/v1/completions
    await asyncio.sleep(0.05)
    return request_id, time.perf_counter() - t0

async def main():
    print("Testing continuous batching with 10 concurrent requests...")
    tasks = [send_simulated_request(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} concurrent requests successfully.")

if __name__ == "__main__":
    asyncio.run(main())
