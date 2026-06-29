from ctxeng.assembly.assembler import ContextAssembler
from ctxeng.eval.benchmark import BenchmarkRunner, format_benchmark_result
from ctxeng.eval.datasets import BUILT_IN_DATASETS, list_datasets
from ctxeng.retrieval.hybrid import HybridRetriever
from ctxeng.stores.memory import InMemoryStore


def main() -> None:
    store = InMemoryStore()
    retriever = HybridRetriever(store)
    assembler = ContextAssembler(store=store)
    runner = BenchmarkRunner(store, retriever, assembler)

    print(f"Running benchmarks on {len(BUILT_IN_DATASETS)} datasets...")
    for name in list_datasets():
        dataset = BUILT_IN_DATASETS[name]
        result = runner.run_dataset(dataset)
        print()
        print(format_benchmark_result(result))
    print()


if __name__ == "__main__":
    main()
