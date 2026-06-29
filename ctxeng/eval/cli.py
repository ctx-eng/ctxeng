"""Eval CLI: benchmark, compare, check."""

import argparse
import json
from dataclasses import asdict
from typing import Dict

from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.eval.benchmark import BenchmarkResult, BenchmarkRunner, format_benchmark_result
from ctxeng.eval.datasets import BUILT_IN_DATASETS, list_datasets
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.stores.memory import InMemoryStore


def _result_to_dict(result: BenchmarkResult) -> dict:
    d = asdict(result)
    d["queries"] = [
        {
            "query": q["query"],
            "latency_ms": q["latency_ms"],
            "precision_at_1": q["precision_at_1"],
            "precision_at_3": q["precision_at_3"],
            "precision_at_5": q["precision_at_5"],
            "recall_at_3": q["recall_at_3"],
            "recall_at_5": q["recall_at_5"],
            "reciprocal_rank": q["reciprocal_rank"],
            "average_precision": q["average_precision"],
            "token_efficiency": q["token_efficiency"],
            "total_tokens": q["total_tokens"],
        }
        for q in d["queries"]
    ]
    return d


def cmd_benchmark(args: argparse.Namespace) -> None:
    store = InMemoryStore()
    retriever = HybridRetriever(store)
    assembler = ContextAssembler(store=store)
    runner = BenchmarkRunner(store, retriever, assembler)

    if args.dataset:
        names = [args.dataset] if args.dataset in BUILT_IN_DATASETS else []
        if not names:
            print(f"Unknown dataset: {args.dataset}. Available: {list_datasets()}")
            return
    else:
        names = list_datasets()

    results: Dict[str, BenchmarkResult] = {}
    print(f"Running benchmarks on {len(names)} datasets...")
    for name in names:
        dataset = BUILT_IN_DATASETS[name]
        result = runner.run_dataset(dataset)
        results[name] = result
        print()
        print(format_benchmark_result(result))
    print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: _result_to_dict(v) for k, v in results.items()}, f, indent=2)
        print(f"Results written to {args.json}")


def cmd_check(args: argparse.Namespace) -> None:
    with open(args.json) as f:
        data = json.load(f)

    thresholds = {}
    for t in args.threshold:
        if "=" not in t:
            print(f"Invalid threshold format: {t} (expected metric=value)")
            return
        key, val = t.split("=", 1)
        thresholds[key] = float(val)

    all_pass = True
    for dataset_name, result in data.items():
        for metric, min_val in thresholds.items():
            actual = result.get(metric)
            if actual is None:
                print(f"  {dataset_name}: {metric} not found in results")
                all_pass = False
                continue
            if actual < min_val:
                print(f"  FAIL {dataset_name}: {metric} = {actual} < {min_val}")
                all_pass = False
            else:
                print(f"  PASS {dataset_name}: {metric} = {actual} >= {min_val}")

    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CtxEng evaluation toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    bm = sub.add_parser("benchmark", help="Run benchmarks")
    bm.add_argument("--dataset", "-d", type=str, default=None, help="Run only the named dataset")
    bm.add_argument("--json", "-j", type=str, default=None, help="Export results as JSON")
    bm.set_defaults(func=cmd_benchmark)

    ck = sub.add_parser("check", help="Check benchmark results against thresholds")
    ck.add_argument("--json", "-j", type=str, required=True, help="Path to benchmark JSON")
    ck.add_argument("--threshold", "-t", type=str, action="append", default=[],
                    help="Threshold in format metric=value")
    ck.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
