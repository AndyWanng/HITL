from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hemorrhage.data.nifti import project_binary_mask
from hemorrhage.pipeline import Pipeline
from hemorrhage.protocol.folds import serpentine_fold_assignment
from hemorrhage.protocol.selection import ScoredCase, build_audit_pool, select_audit, select_routine, split_budget
from tests.helpers import create_synthetic_project, override_project_configs


class ProtocolTests(unittest.TestCase):
    def test_binary_projection(self) -> None:
        raw = np.array([0, 1, 2, 3], dtype=np.int16)
        binary = project_binary_mask(raw)
        np.testing.assert_array_equal(binary, np.array([0, 1, 0, 1], dtype=np.uint8))

    def test_serpentine_fold_assignment(self) -> None:
        items = [(f"case_{idx}", 100 - idx) for idx in range(10)]
        assigned = serpentine_fold_assignment(items, 5)
        self.assertEqual(assigned["case_0"], 1)
        self.assertEqual(assigned["case_4"], 5)
        self.assertEqual(assigned["case_5"], 5)
        self.assertEqual(assigned["case_9"], 1)

    def test_budget_split(self) -> None:
        routine, audit = split_budget(15)
        self.assertEqual(routine, 10)
        self.assertEqual(audit, 5)

    def test_routine_and_audit_selection(self) -> None:
        scored = [
            ScoredCase(case_id=f"c{idx}", fold_id=(idx % 5) + 1, review_count=idx % 3, last_review_round=idx, d=0.5, u=0.5, c=1.0, score=1.0 - idx * 0.05, score_eff=1.0 - idx * 0.05)
            for idx in range(10)
        ]
        routine = select_routine(scored, 3)
        self.assertEqual([item.case_id for item in routine], ["c0", "c1", "c2"])
        audit_pool = build_audit_pool(scored, {item.case_id for item in routine}, 2)
        audit = select_audit(audit_pool, 2, 5)
        self.assertEqual(len(audit), 2)

    def test_pipeline_migrates_old_rounds_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            create_synthetic_project(tmp, num_cases=2)
            override_project_configs(tmp, project_root)
            workspace = tmp / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            db_path = workspace / "state.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE rounds (
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
                """
            )
            conn.commit()
            conn.close()

            Pipeline(project_root=tmp, runtime_config_path=tmp / "configs" / "runtime.local.yaml")

            conn = sqlite3.connect(db_path)
            round_columns = {row[1] for row in conn.execute("PRAGMA table_info(rounds)").fetchall()}
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            conn.close()
            self.assertIn("progress_json", round_columns)
            self.assertIn("review_stats", tables)


if __name__ == "__main__":
    unittest.main()
