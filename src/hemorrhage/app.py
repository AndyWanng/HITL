"""Command-line application."""

from __future__ import annotations

import argparse
from pathlib import Path

from hemorrhage.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hemorrhage", description="Hemorrhage HITL segmentation pipeline")
    parser.add_argument("--project-root", default=".", help="Project root containing configs/, data/, and workspace/")
    parser.add_argument("--runtime-config", default=None, help="Optional runtime yaml path")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-project")
    sub.add_parser("train-round0")

    status = sub.add_parser("status")
    status.add_argument("--round", type=int, dest="round_index")

    plan_round = sub.add_parser("plan-round")
    plan_round.add_argument("--round", type=int, required=True, dest="round_index")
    plan_round.add_argument("--budget", type=int, required=True)

    import_routine = sub.add_parser("import-routine")
    import_routine.add_argument("--round", type=int, required=True, dest="round_index")
    import_routine.add_argument("--input", type=str, required=True, dest="input_dir")

    import_audit_anchor = sub.add_parser("import-audit-anchor")
    import_audit_anchor.add_argument("--round", type=int, required=True, dest="round_index")
    import_audit_anchor.add_argument("--input", type=str, required=True, dest="input_dir")

    import_audit_final = sub.add_parser("import-audit-final")
    import_audit_final.add_argument("--round", type=int, required=True, dest="round_index")
    import_audit_final.add_argument("--input", type=str, required=True, dest="input_dir")

    import_phase1 = sub.add_parser("import-phase1")
    import_phase1.add_argument("--round", type=int, required=True, dest="round_index")
    import_phase1.add_argument("--input", type=str, required=True, dest="input_dir")

    import_phase2 = sub.add_parser("import-phase2")
    import_phase2.add_argument("--round", type=int, required=True, dest="round_index")
    import_phase2.add_argument("--input", type=str, required=True, dest="input_dir")

    finalize_round = sub.add_parser("finalize-round")
    finalize_round.add_argument("--round", type=int, required=True, dest="round_index")

    diagnose_revision = sub.add_parser("diagnose-revision-policy")
    diagnose_revision.add_argument("--round", type=int, required=True, dest="round_index")

    report_round = sub.add_parser("report-round")
    report_round.add_argument("--round", type=int, required=True, dest="round_index")

    predict_external = sub.add_parser("predict-external")
    predict_external.add_argument("--model-tag", default="final")
    predict_external.add_argument("--input-dir", required=True)
    predict_external.add_argument("--output-dir", required=True)

    rebase_paths = sub.add_parser("rebase-paths")
    rebase_paths.add_argument("--from-root", required=True)
    rebase_paths.add_argument("--to-root", required=True)

    normalize_meta = sub.add_parser("normalize-review-metadata")
    normalize_meta.add_argument("--input", required=True)
    normalize_meta.add_argument("--output", required=True)
    normalize_meta.add_argument("--required", nargs="+", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    runtime_config = Path(args.runtime_config).resolve() if args.runtime_config else None
    pipeline = Pipeline(project_root=project_root, runtime_config_path=runtime_config)

    if args.command == "init-project":
        pipeline.init_project()
    elif args.command == "train-round0":
        pipeline.train_round0()
    elif args.command == "status":
        print(pipeline.status(round_index=args.round_index))
    elif args.command == "plan-round":
        pipeline.plan_round(round_index=args.round_index, budget=args.budget)
    elif args.command == "import-routine":
        pipeline.import_routine(round_index=args.round_index, input_dir=Path(args.input_dir))
    elif args.command == "import-audit-anchor":
        pipeline.import_audit_anchor(round_index=args.round_index, input_dir=Path(args.input_dir))
    elif args.command == "import-audit-final":
        pipeline.import_audit_final(round_index=args.round_index, input_dir=Path(args.input_dir))
    elif args.command == "import-phase1":
        pipeline.import_phase1(round_index=args.round_index, input_dir=Path(args.input_dir))
    elif args.command == "import-phase2":
        pipeline.import_phase2(round_index=args.round_index, input_dir=Path(args.input_dir))
    elif args.command == "finalize-round":
        pipeline.finalize_round(round_index=args.round_index)
    elif args.command == "diagnose-revision-policy":
        pipeline.diagnose_revision_policy(round_index=args.round_index)
    elif args.command == "report-round":
        pipeline.report_round(round_index=args.round_index)
    elif args.command == "predict-external":
        pipeline.predict_external(model_tag=args.model_tag, input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    elif args.command == "rebase-paths":
        pipeline.rebase_paths(from_root=Path(args.from_root), to_root=Path(args.to_root))
    elif args.command == "normalize-review-metadata":
        pipeline.normalize_review_metadata(Path(args.input), Path(args.output), args.required)


if __name__ == "__main__":
    main()
