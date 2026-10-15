import time
import numpy as np

class InferenceMetricsTracker:
    def __init__(self):
        self.start_time = None
        self.first_token_time = None
        self.token_times = []

    def start(self):
        self.start_time = time.perf_counter()

    def record_token(self):
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)

    def summary(self):
        ttft = (self.first_token_time - self.start_time) * 1000
        itls = [(self.token_times[i] - self.token_times[i-1]) * 1000 for i in range(1, len(self.token_times))]
        return {
            "TTFT_ms": ttft,
            "ITL_p50_ms": float(np.percentile(itls, 50)) if itls else 0.0,
            "ITL_p95_ms": float(np.percentile(itls, 95)) if itls else 0.0,
            "ITL_p99_ms": float(np.percentile(itls, 99)) if itls else 0.0,
            "Total_tokens": len(self.token_times),
        }

if __name__ == "__main__":
    tracker = InferenceMetricsTracker()
    tracker.start()
    time.sleep(0.08)  # simulate prefill
    tracker.record_token()
    for _ in range(25):
        time.sleep(0.015)  # simulate decode steps
        tracker.record_token()
    print("Metrics Summary:", tracker.summary())
