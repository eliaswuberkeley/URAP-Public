"""Crash-recoverable local writes and checksummed transactional records."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fsync_dir(path.parent)


def write_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def write_csv(path, rows, fields=None):
    fields = fields or sorted({k for r in rows for k in r})
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    atomic_bytes(path, buf.getvalue().encode())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@contextlib.contextmanager
def writer_lock(root):
    path = Path(root) / "runtime/writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pipeline writer is active; do not start a second run") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Store:
    def __init__(self, path, contract):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA fullfsync=ON")
        if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Checkpoint database integrity failure; preserve it for recovery")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (stage TEXT, id TEXT, payload TEXT NOT NULL, checksum TEXT NOT NULL, PRIMARY KEY(stage,id))")
        value = canonical(contract)
        old = self.db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if old and old[0] != value:
            self.db.close()
            raise ValueError("Input/config/code contract changed; cannot mix this run with existing checkpoints")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract',?)", (value,))

    def get(self, stage):
        rows = {}
        for key, payload, checksum in self.db.execute(
                "SELECT id,payload,checksum FROM records WHERE stage=? ORDER BY id", (stage,)):
            if digest(payload.encode()) != checksum:
                raise ValueError(f"Corrupt checkpoint record: {stage}/{key}")
            rows[key] = json.loads(payload)
        return rows

    def put(self, stage, rows):
        with self.db:
            for row in rows:
                payload = canonical(row)
                self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                                (stage, row["image_id"], payload, digest(payload.encode())))

    def backup(self, path):
        path = Path(path)
        tmp = path.with_name(path.name + ".part")
        out = sqlite3.connect(tmp)
        try:
            self.db.backup(out)
        finally:
            out.close()
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)

    def close(self):
        self.db.close()
