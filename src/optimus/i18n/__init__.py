"""Internationalization: JSON string catalogs with dot keys and placeholders.

User-facing strings (DM warnings, embeds, command replies) live in per-locale
JSON files under :mod:`optimus.i18n.locales`. Keys are dotted (``dm.warning``)
and values may contain ``{placeholder}`` fields filled at render time. Lookups
fall back to English when a key or locale is missing, so a partial translation
never produces an empty message.
"""

from __future__ import annotations

from optimus.i18n.catalog import (
    DEFAULT_LOCALE,
    Catalog,
    available_locales,
    get_catalog,
    translate,
)

__all__ = [
    "DEFAULT_LOCALE",
    "Catalog",
    "available_locales",
    "get_catalog",
    "translate",
]
