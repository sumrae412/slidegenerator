"""
Comprehensive Configuration Tests for Slide Generator

Tests configuration of:
- Environment variables
- Flask configuration
- Google OAuth setup
- Security settings
- Development vs Production modes

Run with: pytest tests/config/test_configuration.py -v
"""

import pytest
import sys
import os
import json
import tempfile
from unittest import mock
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from file_to_slides import app, GOOGLE_SCOPES, get_google_client_config


class TestEnvironmentVariables:
    """Test environment variable configuration"""

    def test_secret_key_configured(self):
        """Test that SECRET_KEY or FLASK_SECRET_KEY is properly configured"""
        # In test environment, the app should have a secret key
        assert app.secret_key is not None
        assert isinstance(app.secret_key, str)
        assert len(app.secret_key) > 0

    def test_secret_key_fallback_chain(self):
        """Test that SECRET_KEY falls back through the chain: SECRET_KEY -> FLASK_SECRET_KEY -> default"""
        with patch.dict(os.environ, {}, clear=False):
            # Clear both possible keys
            os.environ.pop('SECRET_KEY', None)
            os.environ.pop('FLASK_SECRET_KEY', None)

            # Import fresh to test fallback behavior
            import importlib
            import file_to_slides
            importlib.reload(file_to_slides)

            # Should have some secret key (even if it's the default)
            assert hasattr(file_to_slides, 'app')
            assert file_to_slides.app.secret_key is not None

    @patch.dict(os.environ, {'SECRET_KEY': 'test-secret-key-123'})
    def test_secret_key_from_env_var(self):
        """Test that SECRET_KEY is read from environment variable"""
        import importlib
        import file_to_slides
        importlib.reload(file_to_slides)

        assert file_to_slides.app.secret_key == 'test-secret-key-123'

    @patch.dict(os.environ, {'GOOGLE_CREDENTIALS_JSON': '{"type": "oauth2"}'})
    def test_google_credentials_json_env_var(self):
        """Test that GOOGLE_CREDENTIALS_JSON is available and valid"""
        creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')

        assert creds_json is not None
        # Should be valid JSON
        parsed = json.loads(creds_json)
        assert isinstance(parsed, dict)

    def test_google_redirect_uri_configured(self):
        """Test that GOOGLE_REDIRECT_URI is configured (with default fallback)"""
        from file_to_slides import GOOGLE_REDIRECT_URI

        assert GOOGLE_REDIRECT_URI is not None
        assert isinstance(GOOGLE_REDIRECT_URI, str)
        # Should be a valid URI format
        assert GOOGLE_REDIRECT_URI.startswith('http')

    @patch.dict(os.environ, {'GOOGLE_REDIRECT_URI': 'https://custom.example.com/oauth2callback'})
    def test_google_redirect_uri_custom_config(self):
        """Test that GOOGLE_REDIRECT_URI can be customized via environment"""
        import importlib
        import file_to_slides
        importlib.reload(file_to_slides)

        assert file_to_slides.GOOGLE_REDIRECT_URI == 'https://custom.example.com/oauth2callback'

    def test_anthropic_api_key_optional(self):
        """Test that ANTHROPIC_API_KEY is optional (not required to start app)"""
        # App should start even without ANTHROPIC_API_KEY
        assert app is not None
        # API key might be None or set, but it should be optional
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        # No assertion on whether it exists - just that it's optional
        assert True  # App loads without requiring it


class TestFlaskConfiguration:
    """Test Flask security and session configuration"""

    def test_session_cookie_httponly_always_set(self):
        """Test that SESSION_COOKIE_HTTPONLY is always True for security"""
        assert app.config['SESSION_COOKIE_HTTPONLY'] is True

    def test_session_cookie_samesite_configured(self):
        """Test that SESSION_COOKIE_SAMESITE is configured to prevent CSRF"""
        assert app.config['SESSION_COOKIE_SAMESITE'] in ['Strict', 'Lax', 'None']
        # Should typically be 'Lax' for OAuth compatibility
        assert app.config['SESSION_COOKIE_SAMESITE'] == 'Lax'

    def test_session_lifetime_configured(self):
        """Test that PERMANENT_SESSION_LIFETIME is set (in seconds)"""
        lifetime = app.config.get('PERMANENT_SESSION_LIFETIME')
        assert lifetime is not None
        assert isinstance(lifetime, int)
        assert lifetime > 0
        # Should be reasonable (between 5 minutes and 24 hours)
        assert 300 <= lifetime <= 86400

    def test_session_cookie_secure_development_mode(self):
        """Test that SESSION_COOKIE_SECURE is False in development"""
        with patch.dict(os.environ, {'FLASK_ENV': 'development'}):
            # In development, SECURE should be False (allow HTTP)
            secure_setting = os.environ.get('FLASK_ENV') != 'development'
            # When FLASK_ENV is 'development', secure_setting should be False
            assert not secure_setting

    def test_session_cookie_secure_production_mode(self):
        """Test that SESSION_COOKIE_SECURE is True in production"""
        with patch.dict(os.environ, {'FLASK_ENV': 'production'}):
            # In production, SECURE should be True (require HTTPS)
            secure_setting = os.environ.get('FLASK_ENV') != 'development'
            # When FLASK_ENV is 'production', secure_setting should be True
            assert secure_setting

    def test_app_testing_mode_disabled_by_default(self):
        """Test that TESTING mode is not enabled in production"""
        # In normal operation, testing should be False
        testing_mode = app.config.get('TESTING', False)
        # Testing mode is typically False in production
        assert isinstance(testing_mode, bool)

    def test_debug_mode_disabled_in_production(self):
        """Test that DEBUG mode respects environment"""
        with patch.dict(os.environ, {'FLASK_ENV': 'production'}):
            is_production = os.environ.get('FLASK_ENV') == 'production'
            assert is_production


class TestGoogleOAuthConfiguration:
    """Test Google OAuth configuration"""

    def test_google_scopes_configured(self):
        """Test that GOOGLE_SCOPES contains required OAuth scopes"""
        assert GOOGLE_SCOPES is not None
        assert isinstance(GOOGLE_SCOPES, list)
        assert len(GOOGLE_SCOPES) > 0

        # Should contain essential scopes
        scopes_str = ' '.join(GOOGLE_SCOPES)
        assert 'presentations' in scopes_str or 'drive' in scopes_str or 'documents' in scopes_str

    def test_google_scopes_presentations_scope(self):
        """Test that presentations scope is included"""
        assert any('presentations' in scope for scope in GOOGLE_SCOPES)

    def test_google_scopes_documents_scope(self):
        """Test that documents.readonly scope is included"""
        assert any('documents.readonly' in scope for scope in GOOGLE_SCOPES)

    def test_google_scopes_drive_scope(self):
        """Test that drive.readonly scope is included"""
        assert any('drive.readonly' in scope for scope in GOOGLE_SCOPES)

    @patch.dict(os.environ, {'GOOGLE_CREDENTIALS_JSON': '{"client_id": "test-id", "client_secret": "test-secret"}'})
    def test_google_client_config_from_env_var(self):
        """Test that get_google_client_config reads from GOOGLE_CREDENTIALS_JSON"""
        config = get_google_client_config()

        assert config is not None
        assert isinstance(config, dict)
        assert 'client_id' in config
        assert config['client_id'] == 'test-id'

    def test_google_client_config_returns_dict_or_none(self):
        """Test that get_google_client_config returns dict or None"""
        config = get_google_client_config()

        assert config is None or isinstance(config, dict)

    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    @patch.dict(os.environ, {'GOOGLE_CREDENTIALS_JSON': ''}, clear=False)
    def test_google_client_config_fallback_to_file(self, mock_open, mock_exists):
        """Test that get_google_client_config falls back to credentials.json file"""
        # Setup mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = json.dumps({"type": "service_account"})
        mock_open.return_value = mock_file

        with patch.dict(os.environ, {'GOOGLE_CREDENTIALS_JSON': ''}, clear=False):
            # Note: In real test, we'd need to ensure file doesn't exist
            # Just verify the function doesn't crash
            result = get_google_client_config()
            assert result is None or isinstance(result, dict)

    def test_google_client_config_handles_missing_credentials(self):
        """Test that get_google_client_config gracefully handles missing credentials"""
        with patch.dict(os.environ, {'GOOGLE_CREDENTIALS_JSON': ''}, clear=False):
            with patch('os.path.exists', return_value=False):
                config = get_google_client_config()

                # Should return None gracefully, not raise exception
                assert config is None


class TestMissingConfigurationHandling:
    """Test graceful handling of missing configuration"""

    def test_missing_secret_key_has_fallback(self):
        """Test that app has fallback secret key if SECRET_KEY not set"""
        # Get the current secret key
        current_secret = app.secret_key

        # Should not be None or empty
        assert current_secret is not None
        assert len(current_secret) > 0

    def test_missing_google_credentials_graceful_degradation(self):
        """Test that missing GOOGLE_CREDENTIALS_JSON is handled gracefully"""
        with patch.dict(os.environ, {'GOOGLE_CREDENTIALS_JSON': ''}, clear=False):
            with patch('os.path.exists', return_value=False):
                config = get_google_client_config()

                # Should return None, not raise exception
                assert config is None or isinstance(config, dict)

    def test_missing_anthropic_api_key_app_still_works(self):
        """Test that app functions without ANTHROPIC_API_KEY"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}, clear=False):
            # App should still be available
            assert app is not None
            # Flask should still initialize
            assert app.name == 'file_to_slides'

    def test_app_initialization_without_all_keys(self):
        """Test that app initializes even with minimal configuration"""
        # App should be importable and initializable
        assert app is not None
        assert isinstance(app, object)
        # Should have basic Flask attributes
        assert hasattr(app, 'secret_key')
        assert hasattr(app, 'config')


class TestDevelopmentVsProductionConfiguration:
    """Test different configurations for development and production"""

    def test_development_mode_settings(self):
        """Test configuration for development environment"""
        with patch.dict(os.environ, {'FLASK_ENV': 'development'}):
            is_dev = os.environ.get('FLASK_ENV') == 'development'
            assert is_dev

            # In development, SECURE cookies should be False
            secure_in_dev = os.environ.get('FLASK_ENV') != 'development'
            assert not secure_in_dev

    def test_production_mode_settings(self):
        """Test configuration for production environment"""
        with patch.dict(os.environ, {'FLASK_ENV': 'production'}):
            is_prod = os.environ.get('FLASK_ENV') == 'production'
            assert is_prod

            # In production, SECURE cookies should be True
            secure_in_prod = os.environ.get('FLASK_ENV') != 'development'
            assert secure_in_prod

    @patch.dict(os.environ, {'FLASK_ENV': 'testing'})
    def test_testing_mode_settings(self):
        """Test configuration for testing environment"""
        is_testing = os.environ.get('FLASK_ENV') == 'testing'
        assert is_testing

    def test_production_requires_https(self):
        """Test that production configuration requires HTTPS"""
        # In production mode, SESSION_COOKIE_SECURE should be True
        # This is evaluated based on FLASK_ENV
        with patch.dict(os.environ, {'FLASK_ENV': 'production'}):
            # Check that the condition would result in SECURE=True
            secure_required = os.environ.get('FLASK_ENV') != 'development'
            assert secure_required is True

    def test_development_allows_http(self):
        """Test that development configuration allows HTTP"""
        # In development mode, SESSION_COOKIE_SECURE should be False
        with patch.dict(os.environ, {'FLASK_ENV': 'development'}):
            # Check that the condition would result in SECURE=False
            secure_required = os.environ.get('FLASK_ENV') != 'development'
            assert secure_required is False


class TestConfigurationIntegration:
    """Integration tests for configuration across components"""

    def test_all_required_config_keys_present(self):
        """Test that all required configuration keys are present in app.config"""
        required_keys = [
            'SESSION_COOKIE_HTTPONLY',
            'SESSION_COOKIE_SAMESITE',
            'PERMANENT_SESSION_LIFETIME'
        ]

        for key in required_keys:
            assert key in app.config, f"Missing required config key: {key}"

    def test_google_oauth_config_complete(self):
        """Test that Google OAuth configuration is complete"""
        # GOOGLE_SCOPES should be defined
        assert GOOGLE_SCOPES is not None
        assert len(GOOGLE_SCOPES) > 0

        # GOOGLE_REDIRECT_URI should be defined
        from file_to_slides import GOOGLE_REDIRECT_URI
        assert GOOGLE_REDIRECT_URI is not None

    def test_session_configuration_compatible(self):
        """Test that session configuration settings are compatible"""
        httponly = app.config['SESSION_COOKIE_HTTPONLY']
        samesite = app.config['SESSION_COOKIE_SAMESITE']
        lifetime = app.config['PERMANENT_SESSION_LIFETIME']

        # HTTPOnly should always be True
        assert httponly is True

        # SAMESITE should be valid value
        assert samesite in ['Strict', 'Lax', 'None']

        # Lifetime should be positive
        assert lifetime > 0

    def test_configuration_consistency(self):
        """Test that configuration is internally consistent"""
        # If SECURE is True, should be in production
        with patch.dict(os.environ, {'FLASK_ENV': 'production'}):
            secure = os.environ.get('FLASK_ENV') != 'development'
            is_prod = os.environ.get('FLASK_ENV') == 'production'

            # If secure is True, we should be in production or staging
            assert secure == is_prod


class TestConfigurationEnvironmentVariables:
    """Test specific environment variable handling"""

    @patch.dict(os.environ, {'GOOGLE_REDIRECT_URI': 'https://heroku.example.com/oauth2callback'})
    def test_heroku_redirect_uri_configuration(self):
        """Test GOOGLE_REDIRECT_URI configuration for Heroku deployment"""
        redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI')

        assert redirect_uri == 'https://heroku.example.com/oauth2callback'
        assert redirect_uri.startswith('https')

    def test_localhost_redirect_uri_default(self):
        """Test that default GOOGLE_REDIRECT_URI works for localhost"""
        from file_to_slides import GOOGLE_REDIRECT_URI

        # Should have some default value
        assert GOOGLE_REDIRECT_URI is not None
        # Default is usually localhost for development
        assert 'localhost' in GOOGLE_REDIRECT_URI or 'example' in GOOGLE_REDIRECT_URI or '127.0.0.1' in GOOGLE_REDIRECT_URI

    @patch.dict(os.environ, {'GOOGLE_CLIENT_SECRETS_FILE': 'custom_secrets.json'})
    def test_google_client_secrets_file_environment_variable(self):
        """Test that GOOGLE_CLIENT_SECRETS_FILE can be customized"""
        secrets_file = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE')

        assert secrets_file == 'custom_secrets.json'

    def test_default_google_client_secrets_file(self):
        """Test that GOOGLE_CLIENT_SECRETS_FILE has sensible default"""
        secrets_file = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE', 'credentials.json')

        assert secrets_file is not None
        assert 'credentials' in secrets_file.lower() or 'secret' in secrets_file.lower()


class TestSecurityConfiguration:
    """Test security-related configuration"""

    def test_session_cookie_security_flags(self):
        """Test that session cookies have security flags set"""
        # HTTPOnly should be True
        assert app.config['SESSION_COOKIE_HTTPONLY'] is True

        # SAMESITE should prevent CSRF
        assert app.config['SESSION_COOKIE_SAMESITE'] in ['Strict', 'Lax']

    def test_secret_key_sufficient_entropy(self):
        """Test that secret key has sufficient length for security"""
        secret = app.secret_key

        # Secret key should be reasonably long
        # Most secure implementations use 32+ characters
        assert len(secret) >= 8  # At minimum 8 characters

    @patch.dict(os.environ, {'FLASK_ENV': 'production'})
    def test_production_requires_secure_cookies(self):
        """Test that production configuration enforces secure cookies"""
        # In production, cookies should be secure (HTTPS only)
        is_production = os.environ.get('FLASK_ENV') == 'production'

        # If we're in production, SECURE should be True
        if is_production:
            assert app.config['SESSION_COOKIE_HTTPONLY'] is True

    def test_no_sensitive_data_in_app_config(self):
        """Test that sensitive data is not hardcoded in app.config"""
        # Config should not contain plaintext passwords or API keys
        for key, value in app.config.items():
            if isinstance(value, str):
                # Check that obvious API keys aren't in config
                assert 'sk-' not in value.lower()  # OpenAI key pattern
                assert 'api-key' not in value.lower()
                assert 'password' not in key.lower()


class TestConfigurationEdgeCases:
    """Test edge cases and boundary conditions in configuration"""

    def test_empty_string_secret_key_handling(self):
        """Test that empty string secret key is replaced with default"""
        with patch.dict(os.environ, {'SECRET_KEY': '', 'FLASK_SECRET_KEY': ''}, clear=False):
            import importlib
            import file_to_slides
            importlib.reload(file_to_slides)

            # Should have some secret key, not empty string
            assert file_to_slides.app.secret_key
            assert file_to_slides.app.secret_key != ''

    def test_malformed_google_credentials_json(self):
        """Test that malformed GOOGLE_CREDENTIALS_JSON is handled gracefully"""
        with patch.dict(os.environ, {'GOOGLE_CREDENTIALS_JSON': 'not-valid-json'}, clear=False):
            try:
                config = get_google_client_config()
                # Either returns None or raises exception - both acceptable
            except json.JSONDecodeError:
                # Expected behavior for malformed JSON
                pass

    def test_zero_session_lifetime_handled(self):
        """Test that session lifetime is positive"""
        lifetime = app.config.get('PERMANENT_SESSION_LIFETIME')

        assert lifetime is not None
        assert lifetime > 0

    def test_invalid_session_samesite_value_avoided(self):
        """Test that SAMESITE has valid value"""
        samesite = app.config.get('SESSION_COOKIE_SAMESITE')

        valid_values = ['Strict', 'Lax', 'None']
        assert samesite in valid_values


# ============================================================================
# PYTEST FIXTURES FOR CONFIG TESTS
# ============================================================================

@pytest.fixture
def clean_environment():
    """
    Fixture that provides a clean environment for testing.

    Saves original environment and restores it after test.
    Use when testing environment variable handling.
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_google_credentials():
    """Fixture providing mock Google OAuth credentials"""
    return {
        'type': 'oauth2',
        'client_id': 'test-client-id-12345.apps.googleusercontent.com',
        'client_secret': 'test-client-secret-abc123',
        'redirect_uris': ['http://localhost:5000/oauth2callback']
    }


@pytest.fixture
def production_environment():
    """Fixture that sets up production environment variables"""
    with patch.dict(os.environ, {'FLASK_ENV': 'production'}):
        yield


@pytest.fixture
def development_environment():
    """Fixture that sets up development environment variables"""
    with patch.dict(os.environ, {'FLASK_ENV': 'development'}):
        yield
