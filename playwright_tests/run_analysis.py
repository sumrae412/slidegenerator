"""Run comprehensive UI analysis and generate report."""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime


def run_tests():
    """Run Playwright tests and collect results."""
    print("🚀 Starting Playwright UI Analysis...\n")

    # Run all tests
    result = subprocess.run(
        ['pytest', '-v', '--tb=short'],
        cwd=Path(__file__).parent.parent,
        capture_output=False
    )

    return result.returncode


def generate_report():
    """Generate comprehensive UI analysis report."""
    report_path = Path(__file__).parent / 'ui_analysis_report.md'
    screenshots_dir = Path(__file__).parent / 'screenshots'

    # Load analysis files if they exist
    complexity_path = Path(__file__).parent / 'ui_complexity_analysis.json'
    friction_path = Path(__file__).parent / 'friction_points.json'
    performance_path = Path(__file__).parent / 'performance_metrics.json'
    resource_path = Path(__file__).parent / 'resource_report.json'

    report = [
        "# UI Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
    ]

    # Complexity Analysis
    if complexity_path.exists():
        with open(complexity_path) as f:
            complexity = json.load(f)

        report.extend([
            "## 📊 UI Complexity Analysis\n",
            f"- **Total Buttons**: {complexity['totalButtons']}",
            f"- **Total Input Fields**: {complexity['totalInputs']}",
            f"- **Total Cards**: {complexity['totalCards']}",
            f"- **Collapsible Sections**: {complexity['totalCollapsibleSections']}",
            f"- **Visible Text Length**: {complexity['visibleText']:,} characters",
            f"\n### Heading Structure",
            f"- H1: {complexity['headings']['h1']}",
            f"- H2: {complexity['headings']['h2']}",
            f"- H3: {complexity['headings']['h3']}",
            f"- H4: {complexity['headings']['h4']}",
            "\n"
        ])

    # Friction Points
    if friction_path.exists():
        with open(friction_path) as f:
            friction = json.load(f)

        severity_emoji = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }

        report.extend([
            "## ⚠️  User Friction Points\n",
            f"**Severity**: {severity_emoji.get(friction['severity'], '⚪')} {friction['severity'].upper()}\n",
        ])

        if friction['friction_points']:
            for point in friction['friction_points']:
                report.append(f"- {point}")
        else:
            report.append("- ✅ No major friction points detected")

        report.append("\n")

    # Performance Metrics
    if performance_path.exists():
        with open(performance_path) as f:
            perf = json.load(f)

        report.extend([
            "## ⚡ Performance Metrics\n",
            f"- **Page Load Time**: {perf.get('page_load_time', 0):.2f}s",
            f"- **DOM Content Loaded**: {perf.get('domContentLoaded', 0):.0f}ms",
            f"- **Load Complete**: {perf.get('loadComplete', 0):.0f}ms",
            f"- **DOM Interactive**: {perf.get('domInteractive', 0):.0f}ms",
            f"- **First Paint**: {perf.get('firstPaint', 0):.0f}ms",
            "\n"
        ])

    # Resource Analysis
    if resource_path.exists():
        with open(resource_path) as f:
            resources = json.load(f)

        report.extend([
            "## 📦 Resource Loading\n",
            f"- **Total Resources**: {resources['total_resources']}",
            f"- **Total Size**: {resources['total_size_mb']:.2f} MB",
            f"- **Failed Resources**: {len(resources['failed_resources'])}",
            "\n"
        ])

    # Screenshots
    if screenshots_dir.exists():
        baseline_dir = screenshots_dir / 'baseline'
        if baseline_dir.exists():
            screenshots = list(baseline_dir.glob('*.png'))
            if screenshots:
                report.extend([
                    "## 📸 Visual Regression Screenshots\n",
                    f"Captured {len(screenshots)} baseline screenshots:\n"
                ])
                for screenshot in screenshots:
                    report.append(f"- `{screenshot.name}`")
                report.append("\n")

    # Recommendations
    report.extend([
        "## 💡 Recommendations for UI Streamlining\n",
        "### High Priority",
        "1. **Reduce Cognitive Load**: Collapse advanced settings by default",
        "2. **Progressive Disclosure**: Show only essential options initially",
        "3. **Visual Hierarchy**: Reduce number of cards and borders",
        "4. **Simplify Layout**: Consolidate related settings into single sections",
        "\n### Design Improvements",
        "1. **Modernize Typography**: Increase spacing, reduce text density",
        "2. **Streamline Colors**: Use more white space, fewer background colors",
        "3. **Clearer CTAs**: Make primary actions more prominent",
        "4. **Mobile-First**: Optimize for smaller screens first",
        "\n### Performance Optimizations",
        "1. **Reduce Resource Count**: Bundle CSS/JS where possible",
        "2. **Lazy Load**: Load collapsible section content on-demand",
        "3. **Optimize Images**: Use WebP format for faster loading",
        "\n"
    ])

    # Save report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n📄 Report generated: {report_path}")
    print("\n" + "="*60)
    print("REPORT PREVIEW")
    print("="*60)
    print('\n'.join(report[:50]))  # Print first 50 lines


if __name__ == '__main__':
    exit_code = run_tests()
    print("\n" + "="*60)
    generate_report()
    print("="*60 + "\n")
    sys.exit(exit_code)
