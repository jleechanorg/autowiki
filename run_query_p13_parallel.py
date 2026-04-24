#!/usr/bin/env python3
"""
P13 Parallel Query Runner
Runs multiple queries in parallel using background processes.
Usage: python3 run_query_p13_parallel.py [start_q] [end_q]
Default: Q1-Q5
"""
import subprocess, sys, os, time
from datetime import datetime

LOG_FILE = "/home/jleechan/projects_other/autowiki/benchmark_logs/run_query_p13.log"
SCRIPT = "/home/jleechan/projects_other/autowiki/run_query_p13.py"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_parallel(start_q, end_q):
    log(f"PARALLEL RUN: Starting Q{start_q}-Q{end_q} in parallel")
    procs = []
    for qnum in range(start_q, end_q + 1):
        log(f"  Launching Q{qnum} in background...")
        p = subprocess.Popen(
            ["python3", SCRIPT, str(qnum)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/home/jleechan/projects_other/autowiki"
        )
        procs.append((qnum, p))
        time.sleep(0.5)  # Stagger launches slightly

    log(f"  All {len(procs)} queries launched. Monitoring...")

    # Monitor progress
    while True:
        time.sleep(30)
        # Check which checkpoints exist
        import glob
        checkpoints = sorted(glob.glob("/home/jleechan/projects_other/autowiki/benchmark_logs/checkpoint_p13_q*.json"))
        completed = len(checkpoints)
        log(f"  Progress: {completed}/{end_q - start_q + 1} checkpoints exist")

        # Check if all done
        if completed >= end_q - start_q + 1:
            log(f"  ALL QUERIES COMPLETE: {completed}/{end_q - start_q + 1}")
            break

        # Check for failed processes
        active = 0
        for qnum, p in procs:
            if p.poll() is None:
                active += 1
        log(f"  Active processes: {active}")

        if active == 0 and completed < end_q - start_q + 1:
            log(f"  WARNING: All processes finished but only {completed} checkpoints found")
            break

if __name__ == "__main__":
    start_q = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end_q = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_parallel(start_q, end_q)