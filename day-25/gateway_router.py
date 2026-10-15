from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Production AI Inference Gateway")
BACKENDS = ["http://localhost:8001", "http://localhost:8002"]
worker_request_counts = {b: 0 for b in BACKENDS}

@app.post("/v1/chat/completions")
async def route_completion(request: Request):
    # Least-connections routing
    target_backend = min(worker_request_counts, key=worker_request_counts.get)
    worker_request_counts[target_backend] += 1
    # In live setup: proxy request asynchronously using httpx
    worker_request_counts[target_backend] -= 1
    return JSONResponse({"status": "routed", "target_backend": target_backend})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
