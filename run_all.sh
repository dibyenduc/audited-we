#!/bin/bash
mkdir -p results
uv run python toy_we.py 2>/dev/null | tee results/toy_we.txt
uv run python audited_we.py 100 2>/dev/null | tee results/toy_audit_stepcheck.txt
uv run python cell_audit.py 100 2>/dev/null | tee results/toy_cell_audit.txt
uv run python det_test.py 2>/dev/null | tee results/det_test_run1.txt
uv run python det_test.py 2>/dev/null | tee results/det_test_run2.txt
uv run python exact_truth.py 2>/dev/null | tee results/exact_truth.txt
uv run python real_audit.py 1000 0 2530 2530 2>/dev/null | tee results/real_1000_runs.txt
uv run python real_audit.py 200 30 2530 2530 2>/dev/null | tee results/real_audit_200_30.txt
uv run python audit_strategies.py 30 2530 2>/dev/null | tee results/audit_strategies.txt
uv run python test_irrelevant.py 10 2530 2>/dev/null | tee results/test_irrelevant.txt
