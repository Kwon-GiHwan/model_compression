from model_compression.reporter.base_reporter import BaseReporter


class ConsoleReporter(BaseReporter):
    """
    콘솔에 벤치마크 결과를 출력하는 리포터.
    """

    def report(self, original_result: dict, compressed_result: dict):
        print("\n" + "=" * 60)
        print(" 📊 Benchmark Results")
        print("=" * 60)

        print(f"\n🔷 Original Model")
        self._print_result(original_result)

        print(f"\n🔶 Compressed Model")
        self._print_result(compressed_result)

        print(f"\n📈 Improvement")
        self._print_comparison(original_result, compressed_result)
        print("=" * 60 + "\n")

    def _print_result(self, result: dict):
        print(f"  • Avg Latency: {result['avg_latency_ms']:.3f} ms")
        print(f"  • Min Latency: {result['min_latency_ms']:.3f} ms")
        print(f"  • Max Latency: {result['max_latency_ms']:.3f} ms")
        print(f"  • Total Params: {result['total_params']:,}")
        print(f"  • Model Size: {result['param_size_mb']:.2f} MB")

    def _print_comparison(self, original: dict, compressed: dict):
        if compressed["avg_latency_ms"] > 0:
            speedup = original["avg_latency_ms"] / compressed["avg_latency_ms"]
        else:
            speedup = float("inf")

        if original["param_size_mb"] > 0:
            size_reduction = (1 - compressed["param_size_mb"] / original["param_size_mb"]) * 100
        else:
            size_reduction = 0.0

        if original["total_params"] > 0:
            param_reduction = (1 - compressed["total_params"] / original["total_params"]) * 100
        else:
            param_reduction = 0.0

        print(f"  • Speedup: {speedup:.2f}x")
        print(f"  • Size Reduction: {size_reduction:.2f}%")
        print(f"  • Parameter Reduction: {param_reduction:.2f}%")
