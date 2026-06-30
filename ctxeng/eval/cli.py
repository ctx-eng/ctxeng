"""Eval CLI: benchmark, compare, check."""

import argparse
import json
from dataclasses import asdict

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

    results: dict[str, BenchmarkResult] = {}
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


def cmd_compare(args: argparse.Namespace) -> None:
    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.result) as f:
        result_data = json.load(f)

    print(f"Comparing {args.result} against baseline {args.baseline}")
    print()
    metrics = [
        "avg_precision_at_1",
        "avg_precision_at_3",
        "avg_recall_at_5",
        "mrr",
        "map_score",
        "avg_token_efficiency",
        "avg_latency_ms",
    ]
    all_pass = True
    for dataset_name in result_data:
        if dataset_name not in baseline:
            print(f"  {dataset_name}: not in baseline, skipping")
            continue
        b = baseline[dataset_name]
        r = result_data[dataset_name]
        print(f"  {dataset_name}:")
        for m in metrics:
            bv = b.get(m, 0)
            rv = r.get(m, 0)
            diff = rv - bv
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
            threshold = getattr(args, "regression_threshold", 0.0)
            if diff < -threshold:
                print(f"    {m}: {rv:.4f} vs {bv:.4f} ({diff:+.4f}) {arrow} REGRESSION")
                all_pass = False
            else:
                print(f"    {m}: {rv:.4f} vs {bv:.4f} ({diff:+.4f}) {arrow}")
        print()

    if all_pass:
        print("No regressions detected.")
    else:
        print("Regressions detected — review above.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CtxEng evaluation toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    bm = sub.add_parser("benchmark", help="Run benchmarks")
    bm.add_argument("--dataset", "-d", type=str, default=None, help="Run only the named dataset")
    bm.add_argument("--json", "-j", type=str, default=None, help="Export results as JSON")
    bm.set_defaults(func=cmd_benchmark)

    ck = sub.add_parser("check", help="Check benchmark results against thresholds")
    ck.add_argument("--json", "-j", type=str, required=True, help="Path to benchmark JSON")
    ck.add_argument(
        "--threshold",
        "-t",
        type=str,
        action="append",
        default=[],
        help="Threshold in format metric=value",
    )
    ck.set_defaults(func=cmd_check)

    cp = sub.add_parser("compare", help="Compare results against a baseline")
    cp.add_argument("--baseline", "-b", type=str, required=True, help="Path to baseline JSON")
    cp.add_argument("--result", "-r", type=str, required=True, help="Path to result JSON")
    cp.add_argument(
        "--regression-threshold",
        type=float,
        default=0.05,
        help="Minimum drop to flag as regression (default: 0.05)",
    )
    cp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
