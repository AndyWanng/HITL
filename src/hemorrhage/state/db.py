"""SQLite-backed state store."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hemorrhage.utils import dumps_json, ensure_dir, loads_json


def _row_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@dataclass(slots=True)
class StateStore:
    path: Path

    def connect(self) -> sqlite3.Connection:
        ensure_dir(self.path.parent)
        conn = sqlite3.connect(self.path)
        conn.row_factory = _row_factory
        return conn

    @contextmanager
    def session(self) -> sqlite3.Connection:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.session() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    source_label_path TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    fold_id INTEGER NOT NULL,
                    v0 INTEGER NOT NULL,
                    review_count INTEGER NOT NULL,
                    last_review_round INTEGER NOT NULL,
                    earliest_eligible_round INTEGER NOT NULL,
                    current_raw_label_path TEXT NOT NULL,
                    current_binary_label_path TEXT NOT NULL,
                    current_oof_path TEXT,
                    current_soft_target_path TEXT,
                    current_uncertainty_path TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rounds (
                    round_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    budget INTEGER,
                    non_empty_round_index INTEGER,
                    routine_ids_json TEXT,
                    audit_ids_json TEXT,
                    config_snapshot_json TEXT,
                    metrics_json TEXT,
                    stop_state_json TEXT
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    round_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    label_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (round_index, case_id, phase)
                );
                CREATE TABLE IF NOT EXISTS review_stats (
                    round_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    routine_final_label_path TEXT,
                    audit_anchor_label_path TEXT,
                    audit_final_label_path TEXT,
                    edit_ratio REAL,
                    modified_slices_count REAL,
                    anchor_assisted_dice REAL,
                    review_time REAL,
                    anchor_time REAL,
                    assisted_time REAL,
                    warnings_json TEXT NOT NULL,
                    PRIMARY KEY (round_index, case_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    round_index INTEGER,
                    case_id TEXT,
                    path TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_metrics (
                    round_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    role TEXT,
                    d REAL,
                    u REAL,
                    c REAL,
                    d_bar REAL,
                    u_bar REAL,
                    c_bar REAL,
                    benefit REAL,
                    score REAL,
                    score_eff REAL,
                    PRIMARY KEY (round_index, case_id)
                );
                CREATE TABLE IF NOT EXISTS round_metrics (
                    round_index INTEGER PRIMARY KEY,
                    cov REAL,
                    edit_routine REAL,
                    edit_audit REAL,
                    delta_t REAL,
                    stab_audit REAL,
                    mean_fused_uncertainty REAL,
                    high_uncertainty_fraction REAL,
                    dice_model_final_routine REAL,
                    dice_model_final_audit REAL,
                    routine_median_time REAL,
                    audit_anchor_median_time REAL,
                    audit_assisted_median_time REAL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "rounds", "progress_json", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def upsert_case(self, record: dict[str, Any]) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO cases (
                    case_id, image_path, source_label_path, subject_id, fold_id, v0,
                    review_count, last_review_round, earliest_eligible_round,
                    current_raw_label_path, current_binary_label_path,
                    current_oof_path, current_soft_target_path, current_uncertainty_path,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    image_path=excluded.image_path,
                    source_label_path=excluded.source_label_path,
                    subject_id=excluded.subject_id,
                    fold_id=excluded.fold_id,
                    v0=excluded.v0,
                    review_count=excluded.review_count,
                    last_review_round=excluded.last_review_round,
                    earliest_eligible_round=excluded.earliest_eligible_round,
                    current_raw_label_path=excluded.current_raw_label_path,
                    current_binary_label_path=excluded.current_binary_label_path,
                    current_oof_path=excluded.current_oof_path,
                    current_soft_target_path=excluded.current_soft_target_path,
                    current_uncertainty_path=excluded.current_uncertainty_path,
                    metadata_json=excluded.metadata_json
                """,
                (
                    record["case_id"],
                    record["image_path"],
                    record["source_label_path"],
                    record["subject_id"],
                    record["fold_id"],
                    record["v0"],
                    record["review_count"],
                    record["last_review_round"],
                    record["earliest_eligible_round"],
                    record["current_raw_label_path"],
                    record["current_binary_label_path"],
                    record.get("current_oof_path"),
                    record.get("current_soft_target_path"),
                    record.get("current_uncertainty_path"),
                    dumps_json(record.get("metadata", {})),
                ),
            )

    def get_case(self, case_id: str) -> dict[str, Any]:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(case_id)
        row["metadata"] = loads_json(row.pop("metadata_json"), {})
        return row

    def list_cases(self) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute("SELECT * FROM cases ORDER BY case_id").fetchall()
        for row in rows:
            row["metadata"] = loads_json(row.pop("metadata_json"), {})
        return rows

    def upsert_round(self, round_index: int, record: dict[str, Any]) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO rounds (
                    round_index, status, budget, non_empty_round_index,
                    routine_ids_json, audit_ids_json, config_snapshot_json,
                    metrics_json, stop_state_json, progress_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_index) DO UPDATE SET
                    status=excluded.status,
                    budget=excluded.budget,
                    non_empty_round_index=excluded.non_empty_round_index,
                    routine_ids_json=excluded.routine_ids_json,
                    audit_ids_json=excluded.audit_ids_json,
                    config_snapshot_json=excluded.config_snapshot_json,
                    metrics_json=excluded.metrics_json,
                    stop_state_json=excluded.stop_state_json,
                    progress_json=excluded.progress_json
                """,
                (
                    round_index,
                    record["status"],
                    record.get("budget"),
                    record.get("non_empty_round_index"),
                    dumps_json(record.get("routine_ids")),
                    dumps_json(record.get("audit_ids")),
                    dumps_json(record.get("config_snapshot", {})),
                    dumps_json(record.get("metrics", {})),
                    dumps_json(record.get("stop_state", {})),
                    dumps_json(record.get("progress", {})),
                ),
            )

    def get_round(self, round_index: int) -> dict[str, Any] | None:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM rounds WHERE round_index = ?", (round_index,)).fetchone()
        if row is None:
            return None
        row["routine_ids"] = loads_json(row.pop("routine_ids_json"), [])
        row["audit_ids"] = loads_json(row.pop("audit_ids_json"), [])
        row["config_snapshot"] = loads_json(row.pop("config_snapshot_json"), {})
        row["metrics"] = loads_json(row.pop("metrics_json"), {})
        row["stop_state"] = loads_json(row.pop("stop_state_json"), {})
        row["progress"] = loads_json(row.pop("progress_json"), {})
        return row

    def list_rounds(self) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute("SELECT * FROM rounds ORDER BY round_index").fetchall()
        parsed: list[dict[str, Any]] = []
        for row in rows:
            row["routine_ids"] = loads_json(row.pop("routine_ids_json"), [])
            row["audit_ids"] = loads_json(row.pop("audit_ids_json"), [])
            row["config_snapshot"] = loads_json(row.pop("config_snapshot_json"), {})
            row["metrics"] = loads_json(row.pop("metrics_json"), {})
            row["stop_state"] = loads_json(row.pop("stop_state_json"), {})
            row["progress"] = loads_json(row.pop("progress_json"), {})
            parsed.append(row)
        return parsed

    def upsert_review(self, round_index: int, case_id: str, role: str, phase: str, label_path: str, metadata: dict[str, Any]) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO reviews (round_index, case_id, role, phase, label_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_index, case_id, phase) DO UPDATE SET
                    role=excluded.role,
                    label_path=excluded.label_path,
                    metadata_json=excluded.metadata_json
                """,
                (round_index, case_id, role, phase, label_path, dumps_json(metadata)),
            )

    def list_reviews(self, round_index: int) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE round_index = ? ORDER BY case_id, phase", (round_index,)
            ).fetchall()
        for row in rows:
            row["metadata"] = loads_json(row.pop("metadata_json"), {})
        return rows

    def get_review_stats(self, round_index: int, case_id: str) -> dict[str, Any] | None:
        with self.session() as conn:
            row = conn.execute(
                "SELECT * FROM review_stats WHERE round_index = ? AND case_id = ?",
                (round_index, case_id),
            ).fetchone()
        if row is None:
            return None
        row["warnings"] = loads_json(row.pop("warnings_json"), [])
        return row

    def list_review_stats(self, round_index: int) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                "SELECT * FROM review_stats WHERE round_index = ? ORDER BY case_id",
                (round_index,),
            ).fetchall()
        for row in rows:
            row["warnings"] = loads_json(row.pop("warnings_json"), [])
        return rows

    def upsert_review_stats(self, round_index: int, case_id: str, record: dict[str, Any]) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO review_stats (
                    round_index, case_id, role,
                    routine_final_label_path, audit_anchor_label_path, audit_final_label_path,
                    edit_ratio, modified_slices_count, anchor_assisted_dice,
                    review_time, anchor_time, assisted_time, warnings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_index, case_id) DO UPDATE SET
                    role=excluded.role,
                    routine_final_label_path=excluded.routine_final_label_path,
                    audit_anchor_label_path=excluded.audit_anchor_label_path,
                    audit_final_label_path=excluded.audit_final_label_path,
                    edit_ratio=excluded.edit_ratio,
                    modified_slices_count=excluded.modified_slices_count,
                    anchor_assisted_dice=excluded.anchor_assisted_dice,
                    review_time=excluded.review_time,
                    anchor_time=excluded.anchor_time,
                    assisted_time=excluded.assisted_time,
                    warnings_json=excluded.warnings_json
                """,
                (
                    round_index,
                    case_id,
                    record["role"],
                    record.get("routine_final_label_path"),
                    record.get("audit_anchor_label_path"),
                    record.get("audit_final_label_path"),
                    record.get("edit_ratio"),
                    record.get("modified_slices_count"),
                    record.get("anchor_assisted_dice"),
                    record.get("review_time"),
                    record.get("anchor_time"),
                    record.get("assisted_time"),
                    dumps_json(record.get("warnings", [])),
                ),
            )

    def add_artifact(self, kind: str, path: str, sha256: str, round_index: int | None = None, case_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (kind, round_index, case_id, path, sha256, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, round_index, case_id, path, sha256, dumps_json(metadata or {})),
            )

    def replace_case_metrics(self, round_index: int, rows: Iterable[dict[str, Any]]) -> None:
        with self.session() as conn:
            conn.execute("DELETE FROM case_metrics WHERE round_index = ?", (round_index,))
            conn.executemany(
                """
                INSERT INTO case_metrics (
                    round_index, case_id, role, d, u, c, d_bar, u_bar, c_bar, benefit, score, score_eff
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        round_index,
                        row["case_id"],
                        row.get("role"),
                        row.get("d"),
                        row.get("u"),
                        row.get("c"),
                        row.get("d_bar"),
                        row.get("u_bar"),
                        row.get("c_bar"),
                        row.get("benefit"),
                        row.get("score"),
                        row.get("score_eff"),
                    )
                    for row in rows
                ],
            )

    def upsert_round_metrics(self, round_index: int, metrics: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO round_metrics (
                    round_index, cov, edit_routine, edit_audit, delta_t, stab_audit,
                    mean_fused_uncertainty, high_uncertainty_fraction,
                    dice_model_final_routine, dice_model_final_audit,
                    routine_median_time, audit_anchor_median_time, audit_assisted_median_time,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_index) DO UPDATE SET
                    cov=excluded.cov,
                    edit_routine=excluded.edit_routine,
                    edit_audit=excluded.edit_audit,
                    delta_t=excluded.delta_t,
                    stab_audit=excluded.stab_audit,
                    mean_fused_uncertainty=excluded.mean_fused_uncertainty,
                    high_uncertainty_fraction=excluded.high_uncertainty_fraction,
                    dice_model_final_routine=excluded.dice_model_final_routine,
                    dice_model_final_audit=excluded.dice_model_final_audit,
                    routine_median_time=excluded.routine_median_time,
                    audit_anchor_median_time=excluded.audit_anchor_median_time,
                    audit_assisted_median_time=excluded.audit_assisted_median_time,
                    metadata_json=excluded.metadata_json
                """,
                (
                    round_index,
                    metrics.get("cov"),
                    metrics.get("edit_routine"),
                    metrics.get("edit_audit"),
                    metrics.get("delta_t"),
                    metrics.get("stab_audit"),
                    metrics.get("mean_fused_uncertainty"),
                    metrics.get("high_uncertainty_fraction"),
                    metrics.get("dice_model_final_routine"),
                    metrics.get("dice_model_final_audit"),
                    metrics.get("routine_median_time"),
                    metrics.get("audit_anchor_median_time"),
                    metrics.get("audit_assisted_median_time"),
                    dumps_json(metadata or {}),
                ),
            )
