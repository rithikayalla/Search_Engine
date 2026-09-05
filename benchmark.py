"""
Benchmarking script for indexing and query performance.

Reports:
- Indexing: wall-clock time + peak resident memory, if `--reindex` is passed and ./DEV exists.
- Querying: mean/p50/p95/max latency across TEST_QUERIES (evaluate.py), run against an
  already-built index (final_postings.txt / seek_map.json / doc_id_map.json).

Usage:
    python3 benchmark.py             # benchmark querying only (requires an existing index)
    python3 benchmark.py --reindex   # also rebuild the index from ./DEV and benchmark that
"""

import argparse
import json
import os
import resource
import statistics
import sys
import time

from indexer import Indexer
from search_engine import SearchEngine
from evaluate import TEST_QUERIES


def peak_rss_mb():
    """Peak resident set size of this process so far, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)  # macOS reports bytes
    return peak / 1024  # Linux reports KB


def benchmark_indexing():
    indexer = Indexer()
    if not os.path.isdir(indexer.root_directory):
        print(f"Skipping indexing benchmark: {indexer.root_directory} not found.")
        return None

    print("\n--- Indexing benchmark ---")
    start = time.time()
    report = indexer.run()
    elapsed = time.time() - start
    peak_mb = peak_rss_mb()
    print(f"Indexing time:                {elapsed:.2f}s")
    print(f"Peak memory during indexing:  {peak_mb:.1f} MB")
    return {"elapsed_s": round(elapsed, 2), "peak_rss_mb": round(peak_mb, 1), **report}


def benchmark_queries():
    required = ["final_postings.txt", "seek_map.json", "doc_id_map.json"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"Skipping query benchmark: missing index file(s) {missing}. Run the indexer first.")
        return None

    print("\n--- Query benchmark ---")
    latencies_ms = []
    result_counts = []
    with SearchEngine() as engine:
        for query in TEST_QUERIES:
            start = time.time()
            results = engine.search(query)
            latencies_ms.append((time.time() - start) * 1000)
            result_counts.append(len(results))

    latencies_ms.sort()
    stats = {
        "num_queries": len(latencies_ms),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "p50_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(latencies_ms[max(int(len(latencies_ms) * 0.95) - 1, 0)], 2),
        "max_ms": round(max(latencies_ms), 2),
        "avg_results_per_query": round(statistics.mean(result_counts), 1),
    }

    print(f"Queries run:        {stats['num_queries']}")
    print(f"Mean latency:       {stats['mean_ms']} ms")
    print(f"p50 latency:        {stats['p50_ms']} ms")
    print(f"p95 latency:        {stats['p95_ms']} ms")
    print(f"Max latency:        {stats['max_ms']} ms")
    print(f"Avg results/query:  {stats['avg_results_per_query']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark indexing and query performance.")
    parser.add_argument(
        "--reindex", action="store_true",
        help="Rebuild the index from ./DEV before benchmarking (otherwise only queries are benchmarked).",
    )
    parser.add_argument("--out", default="benchmark_report.json", help="Path to write the JSON summary to.")
    args = parser.parse_args()

    summary = {}
    if args.reindex:
        summary["indexing"] = benchmark_indexing()
    summary["querying"] = benchmark_queries()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
