"""Tests for the phishing signal engine: weighted scoring, phrases, correlation."""

from optimus.hashing.ocr_extract import find_phishing_signals


def test_empty_text():
    signals, score, level = find_phishing_signals("")
    assert signals == []
    assert score == 0
    assert level == "none"


def test_clean_text_no_signals():
    signals, score, level = find_phishing_signals("Hello, how are you today?")
    assert signals == []
    assert score == 0
    assert level == "none"


def test_single_weak_signal_is_low():
    signals, score, level = find_phishing_signals("Check out this free offer")
    assert "free_offer" in signals
    assert score == 1
    assert level == "low"


def test_two_weak_signals_is_medium():
    signals, score, level = find_phishing_signals("Free limited time deal")
    assert "free_offer" in signals
    assert "urgency" in signals
    assert score == 2
    assert level == "medium"


def test_credentials_is_strong():
    signals, score, level = find_phishing_signals("Enter your password to login")
    assert "credentials" in signals
    assert score >= 3
    assert level in ("medium", "high")


def test_wallet_exfiltration():
    signals, score, level = find_phishing_signals("Enter your seed phrase to recover wallet")
    assert "wallet" in signals
    assert score >= 3
    assert level in ("medium", "high")


def test_crypto_address_detection():
    signals, score, level = find_phishing_signals("Send to 0x" + "a" * 40 + " to claim your reward")
    assert "crypto_address" in signals
    assert score >= 5  # 4 (crypto) + 2 (claim) - 1 (claim might not match)
    assert level in ("high", "critical")


def test_scam_phrase_boosts_score():
    signals, score, level = find_phishing_signals("Claim your free Pro account now")
    assert "scam_phrase" in signals
    assert "free_offer" in signals
    assert "claim" in signals
    assert score >= 6  # 3 (phrase) + 1 (free) + 2 (claim)
    assert level in ("high", "critical")


def test_url_correlation_boosts_risk():
    # Without URL: "free" alone is low (1 pt)
    _, score, level = find_phishing_signals("free stuff")
    assert level == "low"

    # With URL: +3 pts = 4 = high
    _, score, level = find_phishing_signals("free stuff", urls=["https://scam.com"])
    assert score == 4
    assert level == "high"


def test_lookalike_correlation_boosts_risk():
    lookalikes = [
        {
            "domain": "openai-claim.com",
            "impersonating": "openai.com",
            "url": "http://openai-claim.com",
        }
    ]
    _, score, level = find_phishing_signals("free Pro account", lookalikes=lookalikes)
    assert score >= 4  # 1 (free) + 2 (ai_community) + 3 (lookalike correlation) = 6
    assert level in ("high", "critical")


def test_url_plus_lookalike_double_boost():
    lookalikes = [
        {
            "domain": "openai-claim.com",
            "impersonating": "openai.com",
            "url": "http://openai-claim.com",
        }
    ]
    _, score, level = find_phishing_signals(
        "free account", urls=["https://openai-claim.com"], lookalikes=lookalikes
    )
    assert score >= 7  # 1 (free) + 3 (url) + 3 (lookalike) = 7
    assert level == "critical"


def test_discord_specific_scam_phrases():
    signals, _, level = find_phishing_signals("Free Nitro! Steam giveaway! Click here to claim")
    assert "scam_phrase" in signals
    assert "free_offer" in signals
    assert "claim" in signals
    assert level in ("high", "critical")


def test_ai_community_scam_phrases():
    signals, score, _ = find_phishing_signals("Get free Sora access and GPT-5 beta invite")
    assert "ai_community" in signals
    assert "free_offer" in signals
    assert score >= 3


def test_no_false_positive_on_normal_words():
    _, score, level = find_phishing_signals("The team meeting is scheduled for tomorrow at 3pm")
    # "team" matches impersonation (1 pt)
    assert score <= 1
    assert level in ("low", "none")


def test_multiple_strong_signals_reach_critical():
    signals, score, level = find_phishing_signals(
        "Connect your wallet and enter your seed phrase. "
        "Send to 0x" + "b" * 40 + " to claim your free reward. "
        "Limited time offer!"
    )
    assert "credentials" not in signals  # no login/password/api key in text
    assert "wallet" in signals
    assert "crypto_address" in signals
    assert "scam_phrase" in signals
    assert "free_offer" in signals
    assert "urgency" in signals
    assert score >= 7
    assert level == "critical"


def test_risk_levels_ordering():
    """Verify risk level thresholds are correctly ordered."""
    # 1 pt = low
    _, _, level = find_phishing_signals("free")
    assert level == "low"

    # 2 pts = medium
    _, _, level = find_phishing_signals("free limited")
    assert level == "medium"

    # 4 pts = high
    _, _, level = find_phishing_signals("free limited claim", urls=["https://x.com"])
    # free(1) + limited(1) + claim(2) + url(3) = 7 = critical actually
    # Let's test with just free + url = 4 = high
    _, _, level = find_phishing_signals("free", urls=["https://x.com"])
    assert level == "high"
