"""
Playwright Configuration for Slide Generator E2E Tests

This configuration file defines settings for running browser-based end-to-end tests
using Playwright with pytest.

Run tests with:
    pytest tests/e2e/test_browser_workflows.py --headed
    pytest tests/e2e/test_browser_workflows.py --browser firefox
    pytest tests/e2e/test_browser_workflows.py --slowmo 1000
"""

import os
from typing import Dict, List
from playwright.sync_api import Browser, BrowserContext, Page

# Test Configuration
TEST_CONFIG = {
    # Base URL for the application (override with PLAYWRIGHT_BASE_URL env var)
    "base_url": os.getenv("PLAYWRIGHT_BASE_URL", "http://localhost:5000"),

    # Browser settings
    "headless": os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true",
    "slow_mo": int(os.getenv("PLAYWRIGHT_SLOW_MO", "0")),  # Milliseconds to slow down operations

    # Timeout settings (milliseconds)
    "timeout": {
        "default": 30000,  # 30 seconds
        "navigation": 60000,  # 60 seconds for page loads
        "api": 120000,  # 2 minutes for API-heavy operations
    },

    # Screenshot settings
    "screenshots": {
        "enabled": True,
        "on_failure": True,
        "directory": "tests/screenshots",
    },

    # Video recording
    "video": {
        "enabled": os.getenv("PLAYWRIGHT_VIDEO", "false").lower() == "true",
        "directory": "tests/videos",
    },

    # Test data
    "test_data": {
        "google_doc_url": "https://docs.google.com/document/d/test123/edit",
        "test_api_key": "sk-ant-test-key-for-e2e-testing",
    },
}

# Browser configurations for cross-browser testing
BROWSER_CONFIGS: List[Dict] = [
    {
        "name": "chromium",
        "use_for_default": True,
        "viewport": {"width": 1280, "height": 720},
    },
    {
        "name": "firefox",
        "use_for_default": False,
        "viewport": {"width": 1280, "height": 720},
    },
    {
        "name": "webkit",
        "use_for_default": False,
        "viewport": {"width": 1280, "height": 720},
    },
]

# Device emulation presets
DEVICE_PRESETS = {
    "desktop": {"width": 1920, "height": 1080},
    "laptop": {"width": 1280, "height": 720},
    "tablet": {"width": 768, "height": 1024},
    "mobile": {"width": 375, "height": 667},
}


def get_browser_context_options(device: str = "desktop") -> Dict:
    """
    Get browser context options for a specific device type.

    Args:
        device: Device type (desktop, laptop, tablet, mobile)

    Returns:
        Dictionary of context options
    """
    viewport = DEVICE_PRESETS.get(device, DEVICE_PRESETS["desktop"])

    options = {
        "viewport": viewport,
        "locale": "en-US",
        "timezone_id": "America/Los_Angeles",
        "permissions": ["clipboard-read", "clipboard-write"],
        "record_video_dir": TEST_CONFIG["video"]["directory"] if TEST_CONFIG["video"]["enabled"] else None,
    }

    return options


def setup_test_context(browser: Browser, device: str = "desktop") -> BrowserContext:
    """
    Create a new browser context with standard test settings.

    Args:
        browser: Playwright browser instance
        device: Device type to emulate

    Returns:
        Configured browser context
    """
    context = browser.new_context(**get_browser_context_options(device))
    context.set_default_timeout(TEST_CONFIG["timeout"]["default"])
    context.set_default_navigation_timeout(TEST_CONFIG["timeout"]["navigation"])

    return context


def take_screenshot(page: Page, name: str, full_page: bool = False) -> str:
    """
    Take a screenshot and save it to the screenshots directory.

    Args:
        page: Playwright page instance
        name: Screenshot filename (without extension)
        full_page: Whether to capture the full scrollable page

    Returns:
        Path to saved screenshot
    """
    if not TEST_CONFIG["screenshots"]["enabled"]:
        return ""

    screenshots_dir = TEST_CONFIG["screenshots"]["directory"]
    os.makedirs(screenshots_dir, exist_ok=True)

    filepath = os.path.join(screenshots_dir, f"{name}.png")
    page.screenshot(path=filepath, full_page=full_page)

    return filepath


# User agent strings for testing
USER_AGENTS = {
    "chrome": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "firefox": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "safari": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "mobile": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
}
