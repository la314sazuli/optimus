"""Catalog loader: per-locale JSON, dotted keys, ``{placeholder}`` formatting.

Catalogs are loaded once from the packaged ``locales/*.json`` files and cached.
Translation resolves a key in the requested locale, then falls back to English,
then to the raw key itself, so a missing translation degrades gracefully rather
than raising at a user-facing boundary.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

DEFAULT_LOCALE = "en"

#: Locale files shipped in the package.
_LOCALE_FILES = {
    "en": "en.json",
    "sr": "sr.json",
}


class Catalog:
    """An immutable, flattened view of one locale's strings."""

    def __init__(self, locale: str, entries: dict[str, str]) -> None:
        self._locale = locale
        self._entries = entries

    @property
    def locale(self) -> str:
        """The locale code this catalog was loaded for."""
        return self._locale

    @property
    def keys(self) -> frozenset[str]:
        """Every dotted key defined in this catalog."""
        return frozenset(self._entries)

    def has(self, key: str) -> bool:
        """Whether ``key`` is defined in this catalog."""
        return key in self._entries

    def get(self, key: str) -> str | None:
        """Return the raw (unformatted) template for ``key``, or ``None``."""
        return self._entries.get(key)


def _flatten(data: dict[str, object], prefix: str = "") -> dict[str, str]:
    """Flatten a nested catalog dict into dotted string keys."""
    out: dict[str, str] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, prefix=f"{dotted}."))
        else:
            out[dotted] = str(value)
    return out


@lru_cache(maxsize=len(_LOCALE_FILES))
def get_catalog(locale: str) -> Catalog:
    """Load (and cache) the catalog for ``locale``, falling back to English.

    An unknown locale yields the English catalog so callers can pass a guild's
    stored locale verbatim without pre-validating it.
    """
    filename = _LOCALE_FILES.get(locale)
    if filename is None:
        if locale == DEFAULT_LOCALE:
            raise KeyError(DEFAULT_LOCALE)  # pragma: no cover - en is always present
        return get_catalog(DEFAULT_LOCALE)
    raw = resources.files("optimus.i18n.locales").joinpath(filename).read_text("utf-8")
    return Catalog(locale, _flatten(json.loads(raw)))


def available_locales() -> tuple[str, ...]:
    """The locale codes with a packaged catalog."""
    return tuple(_LOCALE_FILES)


def translate(key: str, locale: str = DEFAULT_LOCALE, /, **params: object) -> str:
    """Render ``key`` for ``locale`` with ``{placeholder}`` substitution.

    Resolution order: the requested locale, then English, then the key itself.
    Missing placeholders raise :class:`KeyError` (a programming error worth
    surfacing) rather than rendering a half-filled message.
    """
    template = get_catalog(locale).get(key)
    if template is None and locale != DEFAULT_LOCALE:
        template = get_catalog(DEFAULT_LOCALE).get(key)
    if template is None:
        return key
    if not params:
        return template
    return template.format(**params)
