#!/bin/bash
# EvoQAI: Reproducibility Script for Paper 1 & 2

echo "================================================="
echo " EvoQAI: Research Reproducibility Suite"
echo "================================================="

echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

echo "[2/4] Running Unit Tests..."
pytest tests/

echo "[3/4] Running Paper 1 & 2 Core Experiments (Evolution & Causal Twin)..."
python experiments/run_paper1_drift.py

echo "[4/4] Generating Reports..."
echo "All experiments completed successfully. Check experiments/reports/ for generated plots and CSVs."
echo "================================================="
