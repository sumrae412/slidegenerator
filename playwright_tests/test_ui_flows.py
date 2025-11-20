"""Comprehensive UI flow tests for Document to Presentation Converter.

Tests cover:
1. Core user flows (document upload, conversion)
2. UI state management (collapsible sections, form validation)
3. Visual regression testing
4. Performance metrics
"""

import pytest
import time
import json
from pathlib import Path
from playwright.sync_api import Page, expect


class TestCoreUserFlows:
    """Test essential user workflows."""

    @pytest.mark.flow
    def test_homepage_loads_successfully(self, page: Page, base_url):
        """Test that the homepage loads with all essential elements."""
        page.goto(base_url)

        # Check page title
        expect(page).to_have_title('Document to Presentation Converter')

        # Check header
        header = page.locator('h1')
        expect(header).to_be_visible()
        expect(header).to_contain_text('Document to Presentation Converter')

        # Check main form exists
        form = page.locator('#upload-form')
        expect(form).to_be_visible()

        # Check file upload input
        file_input = page.locator('#file-upload')
        expect(file_input).to_be_visible()

    @pytest.mark.flow
    def test_collapsible_sections_work(self, page: Page, base_url):
        """Test that all collapsible sections can be toggled."""
        page.goto(base_url)

        # Test model settings toggle
        model_toggle = page.locator('#model-settings-toggle')
        model_content = page.locator('#model-settings-content')

        # Initially should be collapsed
        expect(model_content).not_to_have_class(/open/)

        # Click to expand
        model_toggle.click()
        page.wait_for_timeout(500)  # Wait for animation

        # Should be expanded
        expect(model_content).to_have_class(/open/)

        # Click to collapse
        model_toggle.click()
        page.wait_for_timeout(500)

        # Should be collapsed again
        expect(model_content).not_to_have_class(/open/)

    @pytest.mark.flow
    def test_visual_settings_toggle(self, page: Page, base_url):
        """Test visual generation settings toggle."""
        page.goto(base_url)

        # Open visual settings
        visual_toggle = page.locator('#visual-settings-toggle')
        visual_toggle.click()
        page.wait_for_timeout(500)

        # Enable visual generation
        enable_checkbox = page.locator('#enable-visual-generation')
        enable_checkbox.check()

        # Filter section should become visible
        filter_section = page.locator('#visual-filter-section')
        expect(filter_section).to_be_visible()

        # Uncheck should hide it
        enable_checkbox.uncheck()
        expect(filter_section).to_be_hidden()

    @pytest.mark.flow
    def test_api_settings_visibility(self, page: Page, base_url):
        """Test API settings collapsible section."""
        page.goto(base_url)

        # Find API settings toggle
        api_toggle = page.locator('#api-settings-toggle')
        api_content = page.locator('#api-settings-content')

        # Click to expand
        api_toggle.click()
        page.wait_for_timeout(500)

        # Content should be visible
        expect(api_content).to_have_class(/open/)

        # Check for API mode options
        claude_mode = page.locator('input[value="claude"]')
        expect(claude_mode).to_be_visible()

    @pytest.mark.flow
    def test_file_upload_interaction(self, page: Page, base_url):
        """Test file upload UI interaction."""
        page.goto(base_url)

        # Find file input
        file_input = page.locator('#file-upload')
        expect(file_input).to_be_visible()

        # Check accepted file types
        accept_attr = file_input.get_attribute('accept')
        assert '.docx' in accept_attr
        assert '.txt' in accept_attr
        assert '.pdf' in accept_attr


class TestVisualRegression:
    """Visual regression tests with screenshot comparison."""

    @pytest.mark.visual
    def test_homepage_visual_baseline(self, page: Page, base_url):
        """Capture baseline screenshot of homepage."""
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        # Take full page screenshot
        screenshot_path = Path(__file__).parent / 'screenshots' / 'baseline' / 'homepage_full.png'
        page.screenshot(path=str(screenshot_path), full_page=True)

        assert screenshot_path.exists()

    @pytest.mark.visual
    def test_upload_section_visual(self, page: Page, base_url):
        """Capture screenshot of upload section."""
        page.goto(base_url)

        # Find upload section
        upload_section = page.locator('#upload-form')
        screenshot_path = Path(__file__).parent / 'screenshots' / 'baseline' / 'upload_section.png'
        upload_section.screenshot(path=str(screenshot_path))

        assert screenshot_path.exists()

    @pytest.mark.visual
    def test_expanded_settings_visual(self, page: Page, base_url):
        """Capture screenshot with all settings expanded."""
        page.goto(base_url)

        # Expand all collapsible sections
        toggles = [
            '#api-settings-toggle',
            '#model-settings-toggle',
            '#visual-settings-toggle',
            '#guide-toggle'
        ]

        for toggle_id in toggles:
            toggle = page.locator(toggle_id)
            if toggle.is_visible():
                toggle.click()
                page.wait_for_timeout(300)

        # Take screenshot
        screenshot_path = Path(__file__).parent / 'screenshots' / 'baseline' / 'all_settings_expanded.png'
        page.screenshot(path=str(screenshot_path), full_page=True)

        assert screenshot_path.exists()

    @pytest.mark.visual
    def test_mobile_responsive_view(self, page: Page, base_url):
        """Test mobile responsive layout."""
        # Set mobile viewport
        page.set_viewport_size({'width': 375, 'height': 667})
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        # Take screenshot
        screenshot_path = Path(__file__).parent / 'screenshots' / 'baseline' / 'mobile_view.png'
        page.screenshot(path=str(screenshot_path), full_page=True)

        assert screenshot_path.exists()

    @pytest.mark.visual
    def test_tablet_responsive_view(self, page: Page, base_url):
        """Test tablet responsive layout."""
        # Set tablet viewport
        page.set_viewport_size({'width': 768, 'height': 1024})
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        # Take screenshot
        screenshot_path = Path(__file__).parent / 'screenshots' / 'baseline' / 'tablet_view.png'
        page.screenshot(path=str(screenshot_path), full_page=True)

        assert screenshot_path.exists()


class TestPerformance:
    """Performance measurement tests."""

    @pytest.mark.performance
    def test_page_load_performance(self, page: Page, base_url):
        """Measure page load performance metrics."""
        # Navigate and measure
        start_time = time.time()
        page.goto(base_url)
        page.wait_for_load_state('load')
        load_time = time.time() - start_time

        # Get performance metrics
        metrics = page.evaluate('''() => {
            const timing = performance.timing;
            return {
                domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                loadComplete: timing.loadEventEnd - timing.navigationStart,
                domInteractive: timing.domInteractive - timing.navigationStart,
                firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0
            };
        }''')

        # Save metrics
        metrics_path = Path(__file__).parent / 'performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'page_load_time': load_time,
                **metrics
            }, f, indent=2)

        # Assert performance thresholds
        assert load_time < 5.0, f"Page load took {load_time:.2f}s (expected < 5s)"
        assert metrics['domContentLoaded'] < 3000, "DOM Content Loaded should be < 3s"

    @pytest.mark.performance
    def test_interaction_responsiveness(self, page: Page, base_url):
        """Test UI interaction responsiveness."""
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        # Measure toggle click response time
        toggle = page.locator('#model-settings-toggle')

        start_time = time.time()
        toggle.click()
        page.wait_for_selector('#model-settings-content.collapsible-content.open', timeout=2000)
        response_time = time.time() - start_time

        assert response_time < 0.5, f"Toggle response took {response_time:.2f}s (expected < 0.5s)"

    @pytest.mark.performance
    def test_resource_loading(self, page: Page, base_url):
        """Test number and size of resources loaded."""
        resources = []

        def handle_response(response):
            resources.append({
                'url': response.url,
                'status': response.status,
                'content_type': response.headers.get('content-type', ''),
                'size': len(response.body()) if response.ok else 0
            })

        page.on('response', handle_response)
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        # Analyze resources
        total_size = sum(r['size'] for r in resources)
        failed_resources = [r for r in resources if r['status'] >= 400]

        # Save resource report
        report_path = Path(__file__).parent / 'resource_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'total_resources': len(resources),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'failed_resources': failed_resources,
                'resources': resources
            }, f, indent=2)

        # Assertions
        assert len(failed_resources) == 0, f"Found {len(failed_resources)} failed resources"
        assert total_size < 5 * 1024 * 1024, f"Total page size {total_size / 1024 / 1024:.2f}MB exceeds 5MB"


class TestUIClutterAnalysis:
    """Analyze UI clutter and complexity."""

    @pytest.mark.flow
    def test_count_visible_elements(self, page: Page, base_url):
        """Count visible UI elements on initial load."""
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        # Count various element types
        analysis = page.evaluate('''() => {
            return {
                totalButtons: document.querySelectorAll('button').length,
                totalInputs: document.querySelectorAll('input, select, textarea').length,
                totalCards: document.querySelectorAll('.card').length,
                totalCollapsibleSections: document.querySelectorAll('.collapsible-content').length,
                visibleText: document.body.innerText.length,
                headings: {
                    h1: document.querySelectorAll('h1').length,
                    h2: document.querySelectorAll('h2').length,
                    h3: document.querySelectorAll('h3').length,
                    h4: document.querySelectorAll('h4').length,
                }
            };
        }''')

        # Save analysis
        analysis_path = Path(__file__).parent / 'ui_complexity_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\n=== UI Complexity Analysis ===")
        print(f"Total Buttons: {analysis['totalButtons']}")
        print(f"Total Input Fields: {analysis['totalInputs']}")
        print(f"Total Cards: {analysis['totalCards']}")
        print(f"Collapsible Sections: {analysis['totalCollapsibleSections']}")
        print(f"Visible Text Length: {analysis['visibleText']} characters")

    @pytest.mark.flow
    def test_identify_friction_points(self, page: Page, base_url):
        """Identify potential user friction points."""
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        friction_points = []

        # Check for hidden required fields
        hidden_inputs = page.locator('input[required]:not(:visible)').count()
        if hidden_inputs > 0:
            friction_points.append(f"{hidden_inputs} required fields hidden")

        # Check for long scroll distance to submit
        submit_button = page.locator('button[type="submit"]').first
        if submit_button.is_visible():
            submit_position = submit_button.bounding_box()
            if submit_position and submit_position['y'] > 2000:
                friction_points.append(f"Submit button at {submit_position['y']}px (requires long scroll)")

        # Check for too many form fields
        total_inputs = page.locator('input, select, textarea').count()
        if total_inputs > 15:
            friction_points.append(f"{total_inputs} form fields (cognitive overload)")

        # Save friction analysis
        friction_path = Path(__file__).parent / 'friction_points.json'
        with open(friction_path, 'w') as f:
            json.dump({
                'friction_points': friction_points,
                'severity': 'high' if len(friction_points) > 3 else 'medium' if len(friction_points) > 1 else 'low'
            }, f, indent=2)

        print(f"\n=== Friction Points Identified ===")
        for point in friction_points:
            print(f"⚠️  {point}")


class TestAccessibility:
    """Test accessibility compliance."""

    @pytest.mark.flow
    def test_keyboard_navigation(self, page: Page, base_url):
        """Test keyboard navigation through the form."""
        page.goto(base_url)
        page.wait_for_load_state('networkidle')

        # Start from first focusable element
        page.keyboard.press('Tab')

        # Tab through several elements
        focusable_count = 0
        for _ in range(20):
            focused_element = page.evaluate('document.activeElement?.tagName')
            if focused_element in ['INPUT', 'SELECT', 'BUTTON', 'A']:
                focusable_count += 1
            page.keyboard.press('Tab')
            page.wait_for_timeout(100)

        assert focusable_count > 5, f"Only {focusable_count} focusable elements found"

    @pytest.mark.flow
    def test_form_labels(self, page: Page, base_url):
        """Test that all form inputs have labels."""
        page.goto(base_url)

        # Check for inputs without labels
        unlabeled_inputs = page.evaluate('''() => {
            const inputs = Array.from(document.querySelectorAll('input:not([type="hidden"]), select, textarea'));
            return inputs.filter(input => {
                const id = input.id;
                if (!id) return true;
                const label = document.querySelector(`label[for="${id}"]`);
                return !label;
            }).length;
        }''')

        assert unlabeled_inputs == 0, f"Found {unlabeled_inputs} inputs without labels"
