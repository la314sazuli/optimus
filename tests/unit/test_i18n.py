"""i18n catalog: locale parity, English fallback, and placeholder formatting."""

from __future__ import annotations

import pytest

from optimus.i18n import DEFAULT_LOCALE, available_locales, get_catalog, translate


def test_en_and_sr_have_identical_key_sets() -> None:
    en = get_catalog("en")
    sr = get_catalog("sr")
    assert en.keys == sr.keys
    assert en.keys  # non-empty


def test_all_locales_share_the_english_key_set() -> None:
    en_keys = get_catalog("en").keys
    for locale in available_locales():
        assert get_catalog(locale).keys == en_keys


def test_unknown_locale_falls_back_to_english() -> None:
    catalog = get_catalog("xx")
    assert catalog.locale == DEFAULT_LOCALE


def test_translate_uses_locale_then_falls_back_to_english() -> None:
    sr = translate("command.no_permission", "sr")
    en = translate("command.no_permission", "en")
    assert sr != en  # genuinely translated, not an English copy
    assert sr.strip()


def test_translate_missing_key_returns_key() -> None:
    assert translate("does.not.exist", "en") == "does.not.exist"


def test_translate_formats_placeholders() -> None:
    rendered = translate("command.hash_added", "en", hash_id="deadbeef")
    assert "deadbeef" in rendered


def test_translate_unknown_locale_falls_back_per_key() -> None:
    # A made-up locale resolves to English for every key.
    assert translate("button.expired", "zz") == translate("button.expired", "en")


def test_sr_placeholders_match_en() -> None:
    # Every placeholder used in en must exist in sr so .format never KeyErrors.
    import re

    en = get_catalog("en")
    sr = get_catalog("sr")
    placeholder = re.compile(r"{(\w+)}")
    for key in en.keys:
        en_params = set(placeholder.findall(en.get(key) or ""))
        sr_params = set(placeholder.findall(sr.get(key) or ""))
        assert en_params == sr_params, f"placeholder mismatch for {key}"


@pytest.mark.parametrize("locale", ["en", "sr"])
def test_known_keys_render(locale: str) -> None:
    assert translate("dm.appeal_button", locale).strip()
