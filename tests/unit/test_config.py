from __future__ import annotations

import pytest

from ad_mri.core.config import Settings
from ad_mri.core.constants import ApiScope, Environment
from ad_mri.core.exceptions import ConfigurationError


def test_get_settings_returns_settings_instance(settings: Settings) -> None:
    assert isinstance(settings, Settings)
    assert settings.app_env == Environment.TEST


def test_database_url_is_assembled_from_components() -> None:
    settings = Settings(database_url=None)
    url = settings.get_database_url()
    assert url.startswith("postgresql+psycopg2://")
    assert settings.postgres_db in url


def test_explicit_database_url_is_used_verbatim() -> None:
    settings = Settings(database_url="postgresql+psycopg2://custom/db")
    assert settings.get_database_url() == "postgresql+psycopg2://custom/db"


def test_prod_requires_non_default_secret_key() -> None:
    with pytest.raises(ConfigurationError):
        Settings(app_env=Environment.PROD, secret_key="dev-secret-key-change-me")


def test_allowed_origins_accepts_comma_separated_string() -> None:
    settings = Settings(allowed_origins="http://a.test, http://b.test")
    assert settings.allowed_origins == ["http://a.test", "http://b.test"]


def test_api_keys_defaults_to_none() -> None:
    assert Settings().api_keys is None


def test_blank_api_key_env_value_means_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 14 regression test: a blank `API_KEY=` env var (the
    .env.example default) must resolve to `None`, matching `api_keys` and
    `inference_model_path`'s identical blank-string convention - see
    `Settings._blank_api_key`."""
    monkeypatch.setenv("API_KEY", "")
    settings = Settings()
    assert settings.api_key is None


def test_api_keys_blank_env_value_means_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 14 regression test: a blank `API_KEYS=` env var must resolve to
    `None`, not raise a startup ValidationError - see
    `Settings._blank_api_keys`'s docstring for the pydantic-settings
    complex-type-JSON-decoding footgun this guards against. (The blank-value
    path only exists for env vars - a direct `Settings(api_keys="")`
    constructor call doesn't go through the same JSON-decoding source, so
    this is deliberately an env-var-driven test, not a constructor-kwarg
    one.)"""
    monkeypatch.setenv("API_KEYS", "")
    settings = Settings()
    assert settings.api_keys is None


def test_api_keys_parses_a_real_json_env_var_into_scoped_enums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEYS", '{"key-a": ["predict"], "key-b": ["predict", "label"]}')
    settings = Settings()
    assert settings.api_keys == {
        "key-a": [ApiScope.PREDICT],
        "key-b": [ApiScope.PREDICT, ApiScope.LABEL],
    }


def test_api_keys_rejects_an_unknown_scope() -> None:
    with pytest.raises(Exception, match="predict.*label"):
        Settings(api_keys={"key-a": ["not-a-real-scope"]})  # type: ignore[dict-item]


def test_api_keys_rejects_an_entry_with_no_scopes() -> None:
    with pytest.raises(ConfigurationError, match="at least one scope"):
        Settings(api_keys={"key-a": []})


def test_prod_accepts_api_keys_in_place_of_the_legacy_single_key() -> None:
    """Requirement 6 must be satisfiable either way - a prod deployment
    using only the multi-key architecture must not be forced to also set
    the legacy single API_KEY it deliberately isn't using."""
    settings = Settings(
        app_env=Environment.PROD,
        secret_key="a-real-production-secret",
        api_key_required=True,
        api_keys={"key-a": ["predict"]},
    )
    assert settings.api_key is None
    assert settings.api_keys is not None


def test_prod_still_requires_some_key_when_api_key_required() -> None:
    with pytest.raises(ConfigurationError, match="API_KEY or API_KEYS"):
        Settings(
            app_env=Environment.PROD,
            secret_key="a-real-production-secret",
            api_key_required=True,
        )
