"""
Playwright Test Fixtures and Configuration

This module provides pytest fixtures for Playwright-based E2E tests.
Fixtures handle browser setup, page initialization, test data, and cleanup.

Usage:
    Tests in this directory automatically get access to these fixtures.
    Simply add parameters like `page`, `context`, or `browser` to test functions.
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Generator
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import configuration
from playwright.config import TEST_CONFIG, setup_test_context, take_screenshot


# ============================================================================
# SESSION-SCOPED FIXTURES (Shared across all tests)
# ============================================================================

@pytest.fixture(scope="session")
def playwright_session() -> Generator[Playwright, None, None]:
    """
    Session-scoped Playwright instance.
    Reused across all tests for better performance.
    """
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser_type_name() -> str:
    """
    Get browser type from pytest option or environment variable.

    Default: chromium
    Override: pytest --browser firefox
    """
    return os.getenv("PLAYWRIGHT_BROWSER", "chromium")


# ============================================================================
# FUNCTION-SCOPED FIXTURES (Fresh for each test)
# ============================================================================

@pytest.fixture
def browser(playwright_session: Playwright, browser_type_name: str) -> Generator[Browser, None, None]:
    """
    Fresh browser instance for each test.

    Args:
        playwright_session: Shared Playwright instance
        browser_type_name: Browser type (chromium, firefox, webkit)

    Yields:
        Browser instance
    """
    # Get browser type
    browser_type = getattr(playwright_session, browser_type_name)

    # Launch browser
    browser_instance = browser_type.launch(
        headless=TEST_CONFIG["headless"],
        slow_mo=TEST_CONFIG["slow_mo"]
    )

    yield browser_instance

    # Cleanup
    browser_instance.close()


@pytest.fixture
def context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """
    Fresh browser context for each test.
    Provides isolation between tests (cookies, storage, etc.).

    Args:
        browser: Browser instance

    Yields:
        BrowserContext instance
    """
    # Create context with standard settings
    ctx = setup_test_context(browser)

    yield ctx

    # Cleanup
    ctx.close()


@pytest.fixture
def page(context: BrowserContext) -> Generator[Page, None, None]:
    """
    Fresh page for each test.

    Args:
        context: Browser context

    Yields:
        Page instance
    """
    page_instance = context.new_page()

    yield page_instance

    # Cleanup
    page_instance.close()


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def base_url() -> str:
    """
    Base URL for the application under test.

    Default: http://localhost:5000
    Override: PLAYWRIGHT_BASE_URL environment variable
    """
    return TEST_CONFIG["base_url"]


@pytest.fixture
def screenshot_on_failure(request, page: Page):
    """
    Automatically capture screenshot on test failure.

    Usage:
        Just include this fixture in your test parameters.
        If test fails, screenshot is saved automatically.
    """
    yield

    # Check if test failed
    if request.node.rep_call.failed:
        test_name = request.node.name
        screenshot_path = take_screenshot(page, f"FAILED_{test_name}")
        print(f"\n📸 Screenshot saved: {screenshot_path}")


@pytest.fixture
def mock_google_doc_url() -> str:
    """
    Mock Google Docs URL for testing.
    """
    return "https://docs.google.com/document/d/1test123abc/edit"


@pytest.fixture
def mock_api_key() -> str:
    """
    Mock API key for testing.
    """
    return "sk-ant-test-key-12345"


@pytest.fixture
def test_data_dir() -> Path:
    """
    Path to test data directory.
    """
    return Path(__file__).parent / "test_data"


# ============================================================================
# PAGE OBJECT FIXTURES
# ============================================================================

@pytest.fixture
def home_page(page: Page, base_url: str):
    """
    Page object for the home page.
    Pre-navigated and ready to use.

    Args:
        page: Playwright page
        base_url: Application base URL

    Returns:
        HomePage object with helper methods
    """
    class HomePage:
        def __init__(self, page: Page, base_url: str):
            self.page = page
            self.base_url = base_url

        def navigate(self):
            """Navigate to home page"""
            self.page.goto(self.base_url)
            return self

        def fill_google_doc_url(self, url: str):
            """Fill in Google Docs URL input"""
            url_input = self.page.locator("input[name*='url'], input[type='url'], input[type='text']").first
            url_input.fill(url)
            return self

        def select_output_format(self, format_type: str = "pptx"):
            """Select output format (pptx or google_slides)"""
            format_radio = self.page.locator(f"input[value*='{format_type}']").first
            if format_radio.count() > 0:
                format_radio.check()
            return self

        def fill_api_key(self, api_key: str):
            """Fill in API key"""
            api_input = self.page.locator("input[name*='api'], input[placeholder*='API']").first
            if api_input.count() > 0:
                api_input.fill(api_key)
            return self

        def toggle_api_settings(self):
            """Toggle API settings visibility"""
            toggle_button = self.page.locator("#api-settings-toggle, button:has-text('API')").first
            if toggle_button.count() > 0:
                toggle_button.click()
                self.page.wait_for_timeout(500)
            return self

        def submit_form(self):
            """Submit the conversion form"""
            submit_button = self.page.locator("button[type='submit'], input[type='submit']").first
            submit_button.click()
            return self

        def wait_for_result(self, timeout: int = 30000):
            """Wait for conversion result"""
            self.page.wait_for_load_state("networkidle", timeout=timeout)
            return self

    # Create and navigate to home page
    hp = HomePage(page, base_url)
    hp.navigate()
    return hp


# ============================================================================
# PYTEST HOOKS
# ============================================================================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to make test result available to fixtures.
    Used for screenshot_on_failure fixture.
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_configure(config):
    """
    Pytest configuration hook.
    Registers custom markers.
    """
    config.addinivalue_line(
        "markers", "playwright: mark test as a Playwright browser test"
    )
    config.addinivalue_line(
        "markers", "visual: mark test as a visual regression test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Setup test environment before running tests.
    Creates necessary directories, etc.
    """
    # Create screenshot directory
    screenshot_dir = Path(TEST_CONFIG["screenshots"]["directory"])
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Create video directory if enabled
    if TEST_CONFIG["video"]["enabled"]:
        video_dir = Path(TEST_CONFIG["video"]["directory"])
        video_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Cleanup after all tests
    # (Optional: clean up old screenshots, videos, etc.)
    pass


# ============================================================================
# BROWSER-SPECIFIC FIXTURES
# ============================================================================

@pytest.fixture
def chromium_browser(playwright_session: Playwright) -> Generator[Browser, None, None]:
    """Chromium-specific browser for cross-browser tests"""
    browser = playwright_session.chromium.launch(headless=TEST_CONFIG["headless"])
    yield browser
    browser.close()


@pytest.fixture
def firefox_browser(playwright_session: Playwright) -> Generator[Browser, None, None]:
    """Firefox-specific browser for cross-browser tests"""
    browser = playwright_session.firefox.launch(headless=TEST_CONFIG["headless"])
    yield browser
    browser.close()


@pytest.fixture
def webkit_browser(playwright_session: Playwright) -> Generator[Browser, None, None]:
    """WebKit-specific browser for cross-browser tests"""
    browser = playwright_session.webkit.launch(headless=TEST_CONFIG["headless"])
    yield browser
    browser.close()


# ============================================================================
# MOBILE EMULATION FIXTURES
# ============================================================================

@pytest.fixture
def mobile_context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """
    Browser context configured for mobile emulation.
    Emulates iPhone 12 viewport.
    """
    ctx = browser.new_context(
        viewport={"width": 390, "height": 844},
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
        is_mobile=True,
        has_touch=True,
    )
    yield ctx
    ctx.close()


@pytest.fixture
def tablet_context(browser: Browser) -> Generator[BrowserContext, None, None]:
    """
    Browser context configured for tablet emulation.
    Emulates iPad Pro viewport.
    """
    ctx = browser.new_context(
        viewport={"width": 1024, "height": 1366},
        user_agent="Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15",
        is_mobile=True,
        has_touch=True,
    )
    yield ctx
    ctx.close()
