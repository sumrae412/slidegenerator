"""
Regression Benchmark - Track quality across versions

Usage:
    # Run benchmark for current version
    python tests/regression_benchmark.py --version v87

    # Compare two versions
    python tests/regression_benchmark.py --compare v86 v87

    # View stored results
    python tests/regression_benchmark.py --list
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_to_slides import DocumentParser
from tests.golden_test_set import GOLDEN_TEST_SET, QUALITY_THRESHOLDS
from tests.quality_metrics import BulletQualityMetrics, format_metrics_report


class RegressionBenchmark:
    """
    Compare quality across different versions
    """

    def __init__(self, results_dir='tests/benchmark_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = BulletQualityMetrics()

    def run_benchmark(self, version_name: str, api_key: str = None):
        """
        Run all test cases and save results

        Args:
            version_name: Version identifier (e.g., 'v87', 'v86_baseline')
            api_key: Claude API key (if None, uses NLP fallback)
        """
        print(f"\n{'=' * 70}")
        print(f"RUNNING BENCHMARK: {version_name}")
        print(f"{'=' * 70}\n")

        parser = DocumentParser(claude_api_key=api_key)
        results = []

        for i, test_case in enumerate(GOLDEN_TEST_SET, 1):
            test_id = test_case['id']
            print(f"[{i}/{len(GOLDEN_TEST_SET)}] Testing: {test_id}...", end=' ')

            try:
                # Generate bullets
                bullets = parser._create_unified_bullets(
                    test_case['input_text'],
                    context_heading=test_case.get('context_heading')
                )

                # Evaluate
                metrics = self.evaluator.evaluate(bullets, test_case)

                # Store result
                result = {
                    'test_id': test_id,
                    'category': test_case.get('category'),
                    'version': version_name,
                    'timestamp': datetime.now().isoformat(),
                    'bullets': bullets,
                    'metrics': metrics,
                    'passed': metrics['overall_quality'] >= QUALITY_THRESHOLDS['overall_quality']
                }
                results.append(result)

                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                print(f"{status} (Quality: {metrics['overall_quality']:.1f})")

            except Exception as e:
                print(f"❌ ERROR: {e}")
                results.append({
                    'test_id': test_id,
                    'version': version_name,
                    'error': str(e),
                    'passed': False
                })

        # Save results
        self._save_results(version_name, results)

        # Print summary
        self._print_summary(version_name, results)

        return results

    def compare_versions(self, version_a: str, version_b: str):
        """
        Generate comparison report between two versions
        """
        print(f"\n{'=' * 70}")
        print(f"COMPARING VERSIONS: {version_a} vs {version_b}")
        print(f"{'=' * 70}\n")

        results_a = self._load_results(version_a)
        results_b = self._load_results(version_b)

        if not results_a:
            print(f"❌ No results found for {version_a}")
            return

        if not results_b:
            print(f"❌ No results found for {version_b}")
            return

        # Calculate aggregate metrics
        metrics_a = self._aggregate_metrics(results_a)
        metrics_b = self._aggregate_metrics(results_b)

        print(f"Overall Quality:")
        print(f"  {version_a}: {metrics_a['avg_overall']:.1f}/100")
        print(f"  {version_b}: {metrics_b['avg_overall']:.1f}/100")
        delta = metrics_b['avg_overall'] - metrics_a['avg_overall']
        delta_pct = (delta / metrics_a['avg_overall']) * 100 if metrics_a['avg_overall'] > 0 else 0
        status = "✅" if delta > 0 else "❌" if delta < -1 else "➖"
        print(f"  Delta: {status} {delta:+.1f} ({delta_pct:+.1f}%)\n")

        print(f"Breakdown:")
        for metric_name in ['structure_score', 'relevance_score', 'style_score', 'readability_score']:
            val_a = metrics_a[metric_name]
            val_b = metrics_b[metric_name]
            delta = val_b - val_a
            status = "✅" if delta > 0 else "❌" if delta < -1 else "➖"
            print(f"  {metric_name.replace('_', ' ').title():<20} "
                  f"{val_a:5.1f} → {val_b:5.1f}  {status} {delta:+5.1f}")

        print(f"\nTest Results:")
        print(f"  Passed: {metrics_a['passed']}/{metrics_a['total']} → "
              f"{metrics_b['passed']}/{metrics_b['total']}")

        # Find regressions
        regressions = self._find_regressions(results_a, results_b)

        if regressions:
            print(f"\n⚠️  REGRESSIONS DETECTED ({len(regressions)} tests):")
            for reg in regressions[:5]:  # Show top 5
                print(f"  - {reg['test_id']}: {reg['quality_a']:.1f} → "
                      f"{reg['quality_b']:.1f} ({reg['delta']:.1f})")

        # Find improvements
        improvements = self._find_improvements(results_a, results_b)

        if improvements:
            print(f"\n✅ IMPROVEMENTS ({len(improvements)} tests):")
            for imp in improvements[:5]:  # Show top 5
                print(f"  - {imp['test_id']}: {imp['quality_a']:.1f} → "
                      f"{imp['quality_b']:.1f} (+{imp['delta']:.1f})")

        print(f"\n{'=' * 70}\n")

    def list_versions(self):
        """List all stored benchmark results"""
        print(f"\n{'=' * 70}")
        print("STORED BENCHMARK RESULTS")
        print(f"{'=' * 70}\n")

        results_files = list(self.results_dir.glob('*.json'))

        if not results_files:
            print("No benchmark results found.")
            return

        for file in sorted(results_files):
            version = file.stem
            data = json.loads(file.read_text())
            timestamp = data.get('timestamp', 'unknown')
            test_count = len(data.get('results', []))
            print(f"  {version:<20} ({timestamp}) - {test_count} tests")

        print(f"\n{'=' * 70}\n")

    def _save_results(self, version_name: str, results: list):
        """Save benchmark results to file"""
        output_file = self.results_dir / f"{version_name}.json"

        data = {
            'version': version_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

        output_file.write_text(json.dumps(data, indent=2))
        print(f"\n✅ Results saved to: {output_file}")

    def _load_results(self, version_name: str):
        """Load benchmark results from file"""
        results_file = self.results_dir / f"{version_name}.json"

        if not results_file.exists():
            return None

        data = json.loads(results_file.read_text())
        return data['results']

    def _aggregate_metrics(self, results: list) -> dict:
        """Calculate aggregate metrics across all tests"""
        passed = sum(1 for r in results if r.get('passed', False))
        total = len(results)

        # Collect all metric values
        overall_scores = []
        structure_scores = []
        relevance_scores = []
        style_scores = []
        readability_scores = []

        for result in results:
            if 'metrics' in result:
                m = result['metrics']
                overall_scores.append(m['overall_quality'])
                structure_scores.append(m['structure_score'])
                relevance_scores.append(m['relevance_score'])
                style_scores.append(m['style_score'])
                readability_scores.append(m['readability_score'])

        return {
            'passed': passed,
            'total': total,
            'avg_overall': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            'structure_score': sum(structure_scores) / len(structure_scores) if structure_scores else 0,
            'relevance_score': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'style_score': sum(style_scores) / len(style_scores) if style_scores else 0,
            'readability_score': sum(readability_scores) / len(readability_scores) if readability_scores else 0
        }

    def _find_regressions(self, results_a: list, results_b: list, threshold: float = 2.0):
        """Find tests that regressed (quality decreased)"""
        regressions = []

        for res_a in results_a:
            test_id = res_a['test_id']
            res_b = next((r for r in results_b if r['test_id'] == test_id), None)

            if res_b and 'metrics' in res_a and 'metrics' in res_b:
                quality_a = res_a['metrics']['overall_quality']
                quality_b = res_b['metrics']['overall_quality']
                delta = quality_b - quality_a

                if delta < -threshold:  # Significant decrease
                    regressions.append({
                        'test_id': test_id,
                        'quality_a': quality_a,
                        'quality_b': quality_b,
                        'delta': delta
                    })

        return sorted(regressions, key=lambda x: x['delta'])

    def _find_improvements(self, results_a: list, results_b: list, threshold: float = 2.0):
        """Find tests that improved (quality increased)"""
        improvements = []

        for res_a in results_a:
            test_id = res_a['test_id']
            res_b = next((r for r in results_b if r['test_id'] == test_id), None)

            if res_b and 'metrics' in res_a and 'metrics' in res_b:
                quality_a = res_a['metrics']['overall_quality']
                quality_b = res_b['metrics']['overall_quality']
                delta = quality_b - quality_a

                if delta > threshold:  # Significant increase
                    improvements.append({
                        'test_id': test_id,
                        'quality_a': quality_a,
                        'quality_b': quality_b,
                        'delta': delta
                    })

        return sorted(improvements, key=lambda x: x['delta'], reverse=True)

    def _print_summary(self, version_name: str, results: list):
        """Print benchmark summary"""
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK SUMMARY: {version_name}")
        print(f"{'=' * 70}")

        passed = sum(1 for r in results if r.get('passed', False))
        total = len(results)

        print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")

        metrics = self._aggregate_metrics(results)
        print(f"\nAverage Scores:")
        print(f"  Overall Quality: {metrics['avg_overall']:.1f}/100")
        print(f"  Structure:       {metrics['structure_score']:.1f}/100")
        print(f"  Relevance:       {metrics['relevance_score']:.1f}/100")
        print(f"  Style:           {metrics['style_score']:.1f}/100")
        print(f"  Readability:     {metrics['readability_score']:.1f}/100")

        # Find failing tests
        failures = [r for r in results if not r.get('passed', False)]
        if failures:
            print(f"\n❌ Failing Tests ({len(failures)}):")
            for failure in failures[:5]:
                test_id = failure['test_id']
                if 'metrics' in failure:
                    quality = failure['metrics']['overall_quality']
                    print(f"  - {test_id}: {quality:.1f}/100")
                else:
                    print(f"  - {test_id}: ERROR")

        print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description='Run regression benchmarks')
    parser.add_argument('--version', help='Version to benchmark (e.g., v87)')
    parser.add_argument('--compare', nargs=2, metavar=('V1', 'V2'),
                       help='Compare two versions')
    parser.add_argument('--list', action='store_true',
                       help='List stored benchmark results')
    parser.add_argument('--api-key', help='Claude API key (optional)')

    args = parser.parse_args()

    benchmark = RegressionBenchmark()

    if args.list:
        benchmark.list_versions()
    elif args.compare:
        benchmark.compare_versions(args.compare[0], args.compare[1])
    elif args.version:
        api_key = args.api_key or os.getenv('CLAUDE_API_KEY')
        benchmark.run_benchmark(args.version, api_key=api_key)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
