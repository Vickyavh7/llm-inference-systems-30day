#!/usr/bin/env bash
echo "=== How to run NVIDIA Nsight Systems Profile ==="
echo "Command:"
echo "  nsys profile --stats=true --trace=cuda,nvtx,osrt -o inference_trace python3 day-12/run_pytorch_profiler.py"
echo ""
echo "Open the resulting 'inference_trace.nsys-rep' in NVIDIA Nsight Systems GUI to view the timeline."
