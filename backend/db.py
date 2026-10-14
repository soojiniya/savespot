"""MongoDB connection helpers for the read-only FastAPI backend."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import dns.resolver
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def configure_srv_resolver_if_needed(uri: str) -> None:
    if not uri.startswith("mongodb+srv://"):
        return
    host = uri.rsplit("@", 1)[-1].split("/", 1)[0].split("?", 1)[0]
    if not host:
        return
    lookup_name = f"_mongodb._tcp.{host}"
    try:
        dns.resolver.resolve(lookup_name, "SRV", lifetime=3)
        return
    except Exception:
        fallback = dns.resolver.Resolver(configure=False)
        fallback.nameservers = ["8.8.8.8", "1.1.1.1"]
        dns.resolver.default_resolver = fallback


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    load_dotenv(ENV_PATH, override=True)
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not configured")
    configure_srv_resolver_if_needed(uri)
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
    )


def get_database() -> Database:
    load_dotenv(ENV_PATH, override=True)
    db_name = os.getenv("MONGODB_DB_NAME")
    if not db_name:
        raise RuntimeError("MONGODB_DB_NAME is not configured")
    return get_client()[db_name]


def close_client() -> None:
    get_client().close()
    get_client.cache_clear()
