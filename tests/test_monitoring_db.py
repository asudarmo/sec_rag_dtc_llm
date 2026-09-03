from monitoring.db import get_connection_params


def test_get_connection_params_uses_defaults(monkeypatch):
    for var in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("monitoring.db.load_dotenv", lambda *a, **k: None)

    params = get_connection_params()

    assert params == {
        "host": "localhost", "port": "5432", "dbname": "sec_rag",
        "user": "postgres", "password": "postgres",
    }


def test_get_connection_params_reads_env_overrides(monkeypatch):
    monkeypatch.setattr("monitoring.db.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
    monkeypatch.setenv("POSTGRES_DB", "custom_db")

    params = get_connection_params()

    assert params["host"] == "db.example.com"
    assert params["dbname"] == "custom_db"
