import subprocess
import time
import csv
import threading
import torch

def background_stress():
    print("Starting background GPU stress workload...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    end_time = time.time() + 6
    while time.time() < end_time:
        _ = torch.matmul(x, x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    print("Stress workload finished.")

def log_telemetry(duration_sec=8, interval_sec=0.5, output_file="gpu_telemetry.csv"):
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ]
    print(f"Logging telemetry to {output_file} every {interval_sec}s...")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gpu_util_pct", "mem_util_pct", "mem_used_mb", "mem_total_mb", "temp_c", "power_w"])
        end_time = time.time() + duration_sec
        while time.time() < end_time:
            try:
                line = subprocess.check_output(query_cmd).decode("utf-8").strip()
                row = [x.strip() for x in line.split(",")]
                writer.writerow(row)
            except Exception as e:
                writer.writerow([time.time(), 0, 0, 0, 0, 0, 0])
            time.sleep(interval_sec)
    print("Telemetry logging complete.")

if __name__ == "__main__":
    t_stress = threading.Thread(target=background_stress)
    t_stress.start()
    log_telemetry(duration_sec=7, interval_sec=0.5)
    t_stress.join()
