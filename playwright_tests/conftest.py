"""Playwright test configuration and fixtures."""

import pytest
import os
import json
from pathlib import Path
from datetime import datetime

# Test configuration
BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')
SCREENSHOTS_DIR = Path(__file__).parent / 'screenshots'
BASELINE_DIR = SCREENSHOTS_DIR / 'baseline'
CURRENT_DIR = SCREENSHOTS_DIR / 'current'
DIFF_DIR = SCREENSHOTS_DIR / 'diff'

# Ensure directories exist
for dir_path in [SCREENSHOTS_DIR, BASELINE_DIR, CURRENT_DIR, DIFF_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope='session')
def base_url():
    """Base URL for the application."""
    return BASE_URL


@pytest.fixture(scope='function')
def screenshot_path(request):
    """Generate screenshot path for the current test."""
    test_name = request.node.name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return CURRENT_DIR / f"{test_name}_{timestamp}.png"


@pytest.fixture(scope='function')
def performance_metrics():
    """Store performance metrics for the test."""
    metrics = {
        'page_load_time': None,
        'time_to_interactive': None,
        'total_requests': 0,
        'failed_requests': 0,
        'dom_content_loaded': None,
    }
    return metrics


@pytest.fixture(scope='session')
def browser_context_args(browser_context_args):
    """Configure browser context with device emulation."""
    return {
        **browser_context_args,
        'viewport': {
            'width': 1920,
            'height': 1080,
        },
        'device_scale_factor': 1,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        'markers', 'slow: marks tests as slow (deselect with -m "not slow")'
    )
    config.addinivalue_line(
        'markers', 'visual: marks tests that capture screenshots for visual regression'
    )
    config.addinivalue_line(
        'markers', 'performance: marks tests that measure performance metrics'
    )
    config.addinivalue_line(
        'markers', 'flow: marks tests that test complete user flows'
    )
