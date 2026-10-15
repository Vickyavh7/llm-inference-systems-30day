#!/usr/bin/env bash
set -e

echo "=== Initializing 30-Day LLM Inference Systems Environment ==="

python3 --version

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt || echo "Note: Run in your environment with internet access."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not detected in current shell."
fi

echo "=== Ready to publish to GitHub ==="
echo "  git init"
echo "  git branch -M main"
echo "  git add ."
echo "  git commit -m 'feat: initial commit for 30-day LLM inference systems roadmap'"
echo "  gh repo create llm-inference-systems-30day --public --source=. --push"
