"""
Unit tests for AffectSense AI Access Key and Session Security.
"""

import time
import pytest
from pathlib import Path
from engine.auth import AuthManager

@pytest.fixture
def auth_mgr(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AUTH_ENABLED=true\n"
        "SESSION_EXPIRATION_MINUTES=0.05\n"  # 3 seconds for test
        "ALLOW_PUBLIC_KEY_GENERATION=true\n"
        "KEY_GENERATION_COOLDOWN_SECONDS=1\n"
        "ADMIN_MASTER_KEY=test_master_secret\n"
        "AUTH_SIGNING_SECRET=test_signature_secret_key\n"
        "MAX_STORED_KEYS=50\n"
    )
    storage_file = tmp_path / ".access_keys.json"
    mgr = AuthManager(env_path=str(env_file), storage_path=str(storage_file))
    return mgr

def test_key_generation_format(auth_mgr):
    success, key, msg = auth_mgr.generate_key("client_1")
    assert success is True
    assert key.startswith("AFTS-")
    parts = key.split("-")
    assert len(parts) == 5  # AFTS-XXXX-XXXX-XXXX-XXXX
    for p in parts[1:]:
        assert len(p) == 4

def test_key_activation_single_use(auth_mgr):
    success, key, _ = auth_mgr.generate_key("client_2")
    assert success is True
    
    # First activation succeeds
    ok, msg, session = auth_mgr.validate_and_activate(key)
    assert ok is True
    assert session is not None
    assert session["is_master"] is False
    assert session["expires_at"] > time.time()
    
    # Second activation fails (single-use constraint)
    ok2, msg2, session2 = auth_mgr.validate_and_activate(key)
    assert ok2 is False
    assert "already been redeemed" in msg2 or "expired" in msg2

def test_invalid_key(auth_mgr):
    ok, msg, session = auth_mgr.validate_and_activate("AFTS-INVALID-KEY-XXXX")
    assert ok is False
    assert "Invalid Access Key" in msg

def test_session_validity_and_expiration(auth_mgr):
    success, key, _ = auth_mgr.generate_key("client_3")
    ok, msg, session = auth_mgr.validate_and_activate(key)
    assert ok is True
    
    # Session is valid immediately
    valid, rem, status = auth_mgr.is_session_valid(session)
    assert valid is True
    assert rem > 0
    
    # Wait for test session duration (3 seconds)
    time.sleep(3.2)
    valid_after, rem_after, status_after = auth_mgr.is_session_valid(session)
    assert valid_after is False
    assert rem_after == 0.0
    assert "expired" in status_after.lower()

def test_master_admin_key(auth_mgr):
    ok, msg, session = auth_mgr.validate_and_activate("test_master_secret")
    assert ok is True
    assert session["is_master"] is True
    valid, rem, _ = auth_mgr.is_session_valid(session)
    assert valid is True
    
    # Master key can be activated repeatedly
    ok2, _, _ = auth_mgr.validate_and_activate("test_master_secret")
    assert ok2 is True

def test_rate_limit_cooldown(auth_mgr):
    ok1, key1, _ = auth_mgr.generate_key("client_cooldown")
    assert ok1 is True
    
    # Immediate second attempt should hit cooldown (cooldown = 1 sec)
    ok2, key2, msg2 = auth_mgr.generate_key("client_cooldown")
    assert ok2 is False
    assert "wait" in msg2.lower()
    
    # After cooldown, allowed again
    time.sleep(1.1)
    ok3, key3, _ = auth_mgr.generate_key("client_cooldown")
    assert ok3 is True

def test_cryptographic_signed_token_tampering(auth_mgr):
    token = auth_mgr.mint_signed_token(duration_minutes=15)
    is_valid, dur, _ = auth_mgr.verify_signed_token(token)
    assert is_valid is True
    assert dur == 15

    # Tamper with the last character
    last_char = token[-1]
    replacement = "A" if last_char != "A" else "B"
    tampered_token = token[:-1] + replacement
    is_tampered_valid, _, err = auth_mgr.verify_signed_token(tampered_token)
    assert is_tampered_valid is False
    assert "signature verification failed" in err.lower()

def test_cloner_environment_default(tmp_path, monkeypatch):
    # Ensure no lingering env vars from other tests
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("ALLOW_PUBLIC_KEY_GENERATION", raising=False)
    monkeypatch.delenv("SESSION_EXPIRATION_MINUTES", raising=False)

    # Simulates a freshly cloned repository with no .env
    empty_dir = tmp_path / "cloned_repo"
    empty_dir.mkdir()
    cloner_mgr = AuthManager(
        env_path=str(empty_dir / ".env"),
        storage_path=str(empty_dir / ".access_keys.json")
    )
    # Auth must be ENABLED by default
    assert cloner_mgr.auth_enabled is True
    # Public generation must be DISABLED by default
    assert cloner_mgr.allow_public_generation is False

def test_cross_machine_token_redemption(tmp_path):
    # Owner machine
    owner_env = tmp_path / "owner.env"
    owner_env.write_text(
        "AUTH_SIGNING_SECRET=shared_production_secret\n"
        "SESSION_EXPIRATION_MINUTES=10\n"
    )
    owner_mgr = AuthManager(env_path=str(owner_env), storage_path=str(tmp_path / "owner_keys.json"))
    issued_key = owner_mgr.mint_signed_token(duration_minutes=10)

    # Cloner machine (has empty key storage, no access to owner keys)
    cloner_env = tmp_path / "cloner.env"
    cloner_env.write_text(
        "AUTH_SIGNING_SECRET=shared_production_secret\n"
    )
    cloner_mgr = AuthManager(env_path=str(cloner_env), storage_path=str(tmp_path / "cloner_keys.json"))

    # Cloner activates key
    ok, msg, session = cloner_mgr.validate_and_activate(issued_key)
    assert ok is True
    assert session["duration_minutes"] == 10
    
    # Cloner cannot reuse same key
    ok2, msg2, _ = cloner_mgr.validate_and_activate(issued_key)
    assert ok2 is False
    assert "already been redeemed" in msg2 or "expired" in msg2


def test_user_details_validation(auth_mgr):
    # Test valid details
    ok, err = auth_mgr.validate_user_details("Alice Smith", "+1234567890", "alice@example.com")
    assert ok is True
    assert err == ""

    # Test short name
    ok, err = auth_mgr.validate_user_details("A", "+1234567890", "alice@example.com")
    assert ok is False
    assert "name" in err.lower()

    # Test invalid email
    ok, err = auth_mgr.validate_user_details("Alice Smith", "+1234567890", "invalid-email")
    assert ok is False
    assert "email" in err.lower()

    # Test short phone
    ok, err = auth_mgr.validate_user_details("Alice Smith", "123", "alice@example.com")
    assert ok is False
    assert "phone" in err.lower()


def test_admin_key_generation_and_activation(auth_mgr):
    # Generate admin key
    success, admin_key, _ = auth_mgr.generate_key(client_id="test_admin", is_admin=True, label="Test Admin")
    assert success is True
    assert auth_mgr.inspect_signed_token(admin_key)["is_admin"] is True

    # Activate admin key without user info (optional for admin)
    ok, msg, session = auth_mgr.validate_and_activate(admin_key, user_info=None, require_user_info=True)
    assert ok is True
    assert session["is_admin"] is True
    assert session["role"] == "admin"

    # Admin key can be activated multiple times
    ok2, _, session2 = auth_mgr.validate_and_activate(admin_key, user_info={"name": "Sub-Admin", "phone": "", "email": ""})
    assert ok2 is True
    assert session2["is_admin"] is True


def test_key_listing_and_revocation(auth_mgr):
    # Generate keys
    auth_mgr.generate_key("u1", role="user", label="User One")
    auth_mgr.generate_key("a1", role="admin", label="Admin One", is_admin=True)

    metrics = auth_mgr.get_admin_dashboard_metrics()
    assert metrics["total_keys"] >= 2
    assert metrics["admin_keys"] >= 1

    keys = auth_mgr.list_keys()
    assert len(keys) >= 2
    
    # Revoke a key
    key_to_revoke = keys[0]["key"]
    revoked = auth_mgr.revoke_key(key_to_revoke)
    assert revoked is True

    # Attempting to activate revoked key must fail
    ok, msg, _ = auth_mgr.validate_and_activate(key_to_revoke)
    assert ok is False
    assert "revoked" in msg.lower()

