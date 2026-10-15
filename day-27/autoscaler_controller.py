def calculate_desired_replicas(current_replicas, queued_requests, target_queue_per_worker=5, max_replicas=10):
    if queued_requests == 0:
        return max(1, current_replicas - 1)
    desired = (queued_requests + target_queue_per_worker - 1) // target_queue_per_worker
    return min(max_replicas, max(1, desired))

if __name__ == "__main__":
    for q in [0, 2, 8, 25, 60]:
        print(f"Queued Requests: {q:<4} -> Desired Worker Replicas: {calculate_desired_replicas(2, q)}")
