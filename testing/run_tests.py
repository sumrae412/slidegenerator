#!/usr/bin/env python3
"""
Test runner for slide generator application
Runs comprehensive tests and generates reports
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def install_dependencies():
    """Install test dependencies"""
    print("Installing test dependencies...")
    return run_command([
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements_test.txt'
    ], "Installing test dependencies")

def run_unit_tests():
    """Run unit tests"""
    return run_command([
        sys.executable, '-m', 'pytest', 'unit/', 
        '-v', '--tb=short', '-m', 'not slow'
    ], "Unit tests")

def run_integration_tests():
    """Run integration tests"""
    return run_command([
        sys.executable, '-m', 'pytest', 'integration/', 
        '-v', '--tb=short'
    ], "Integration tests")

def run_performance_tests():
    """Run performance tests"""
    return run_command([
        sys.executable, '-m', 'pytest', 
        '-v', '--tb=short', '-m', 'performance'
    ], "Performance tests")

def run_all_tests():
    """Run all tests with coverage"""
    return run_command([
        sys.executable, '-m', 'pytest', 
        '-v', '--tb=short', 
        '--cov=../file_to_slides_enhanced',
        '--cov-report=html:reports/coverage',
        '--cov-report=term-missing',
        '--html=reports/test_report.html',
        '--self-contained-html'
    ], "All tests with coverage")

def run_specific_test(test_path):
    """Run a specific test"""
    return run_command([
        sys.executable, '-m', 'pytest', test_path, '-v', '--tb=long'
    ], f"Specific test: {test_path}")

def generate_summary_report(results):
    """Generate a summary report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
Test Execution Summary
Generated: {timestamp}

Test Results:
{'='*50}
"""
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        report += f"{test_name:<30} {status}\n"
    
    report += f"\n{'='*50}\n"
    report += f"Total Tests: {total_tests}\n"
    report += f"Passed: {passed_tests}\n"
    report += f"Failed: {failed_tests}\n"
    report += f"Success Rate: {(passed_tests/total_tests*100):.1f}%\n"
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    with open('reports/summary.txt', 'w') as f:
        f.write(report)
    
    print(report)
    return passed_tests == total_tests

def main():
    parser = argparse.ArgumentParser(description='Run slide generator tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--install-deps', action='store_true', help='Install test dependencies')
    parser.add_argument('--test', type=str, help='Run specific test file')
    parser.add_argument('--no-deps', action='store_true', help='Skip dependency installation')
    
    args = parser.parse_args()
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    results = {}
    
    # Install dependencies unless skipped
    if not args.no_deps:
        print("Checking and installing dependencies...")
        dep_result = install_dependencies()
        results['Dependencies'] = dep_result
        
        if not dep_result:
            print("⚠️  Dependency installation failed. Some tests may not work.")
    
    # Run specific test
    if args.test:
        results[f'Test: {args.test}'] = run_specific_test(args.test)
    
    # Run test suites
    elif args.unit:
        results['Unit Tests'] = run_unit_tests()
    
    elif args.integration:
        results['Integration Tests'] = run_integration_tests()
    
    elif args.performance:
        results['Performance Tests'] = run_performance_tests()
    
    elif args.all:
        results['All Tests'] = run_all_tests()
    
    else:
        # Default: run unit and integration tests
        print("Running default test suite (unit + integration)...")
        results['Unit Tests'] = run_unit_tests()
        results['Integration Tests'] = run_integration_tests()
    
    # Generate summary
    overall_success = generate_summary_report(results)
    
    if overall_success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the reports for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()