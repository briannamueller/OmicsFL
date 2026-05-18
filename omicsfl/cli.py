"""OmicsFL command-line interface."""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="omicsfl",
        description="Download, preprocess, and partition TCGA data for "
                    "federated learning experiments.",
    )
    subparsers = parser.add_subparsers(dest="command")

    gen = subparsers.add_parser(
        "generate",
        help="Generate a federated partition (download + preprocess + split).",
    )
    gen.add_argument("--cohorts", type=str, default="all",
                     help="Comma-separated cohort names, or 'all' (default: all)")
    gen.add_argument("--task", type=str, default="cancer_type",
                     choices=["cancer_type", "stage", "survival"],
                     help="Classification task (default: cancer_type)")
    gen.add_argument("--source", type=str, default="xena",
                     choices=["xena", "gdc"],
                     help="Download backend (default: xena)")
    gen.add_argument("--partition", type=str, default="natural",
                     choices=["natural", "by_cohort", "dirichlet", "pathological"],
                     help="Partition strategy (default: natural)")
    gen.add_argument("--top-genes", type=int, default=2000,
                     help="Top-K genes by variance. 0 = all (default: 2000)")
    gen.add_argument("--min-samples", type=int, default=20,
                     help="Min samples per client (default: 20)")
    gen.add_argument("--train-ratio", type=float, default=0.8,
                     help="Train fraction (default: 0.8)")
    gen.add_argument("--val-ratio", type=float, default=0.0,
                     help="Validation fraction (default: 0.0)")
    gen.add_argument("--seed", type=int, default=42,
                     help="Random seed (default: 42)")
    gen.add_argument("--output-dir", type=str, default=None,
                     help="Output directory (auto-generated if omitted)")
    gen.add_argument("--cache-dir", type=str, default=None,
                     help="Raw download cache directory")
    gen.add_argument("--num-clients", type=int, default=10,
                     help="Number of clients for artificial partitions (default: 10)")
    gen.add_argument("--alpha", type=float, default=0.5,
                     help="Dirichlet concentration (default: 0.5)")
    gen.add_argument("--survival-threshold-days", type=int, default=365 * 3,
                     help="Survival cutoff in days (default: 1095 = 3 years)")
    gen.add_argument("--classes-per-client", type=int, default=2,
                     help="Classes per client for pathological (default: 2)")

    dl = subparsers.add_parser(
        "download",
        help="Download raw TCGA data without partitioning.",
    )
    dl.add_argument("--cohorts", type=str, default="all",
                    help="Comma-separated cohort names, or 'all'")
    dl.add_argument("--source", type=str, default="xena",
                    choices=["xena", "gdc"])
    dl.add_argument("--output-dir", type=str, default=None,
                    help="Output directory")

    subparsers.add_parser(
        "info",
        help="List available cohorts and tasks.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "download":
        _cmd_download(args)
    elif args.command == "info":
        _cmd_info()


def _cmd_generate(args):
    from omicsfl.generate import generate

    cohorts = args.cohorts if args.cohorts == "all" else args.cohorts.split(",")

    path = generate(
        cohorts=cohorts,
        task=args.task,
        source=args.source,
        partition=args.partition,
        top_genes=args.top_genes,
        min_samples=args.min_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        survival_threshold_days=args.survival_threshold_days,
        num_clients=args.num_clients,
        alpha=args.alpha,
        classes_per_client=args.classes_per_client,
    )
    print(f"\nPartition ready at: {path}")


def _cmd_download(args):
    from omicsfl.download.registry import TCGA_COHORTS, validate_cohorts

    if args.cohorts == "all":
        cohorts = list(TCGA_COHORTS)
    else:
        cohorts = validate_cohorts(args.cohorts.split(","))

    output_dir = args.output_dir or f"./omicsfl_cache/{args.source}"

    if args.source == "xena":
        from omicsfl.download.xena import download
        download(cohorts, output_dir=output_dir)
    elif args.source == "gdc":
        from omicsfl.download.gdc import download
        download(cohorts, output_dir=output_dir)

    print("\nDownload complete.")


def _cmd_info():
    from omicsfl.download.registry import TCGA_COHORTS

    print("OmicsFL — Available TCGA Cohorts")
    print("=" * 50)
    for i, c in enumerate(TCGA_COHORTS, 1):
        print(f"  {i:2d}. TCGA-{c}")
    print(f"\n  Total: {len(TCGA_COHORTS)} cohorts")
    print()
    print("Classification Tasks")
    print("=" * 50)
    print("  cancer_type  — Multi-class cancer type (requires 2+ cohorts)")
    print("  stage        — Binary: advanced vs non-advanced")
    print("  survival     — Binary: survived past threshold vs deceased (Xena only)")
    print()
    print("Partition Strategies")
    print("=" * 50)
    print("  natural      — By contributing hospital (real-world heterogeneity)")
    print("  by_cohort    — Each cancer type = one client")
    print("  dirichlet    — Synthetic non-IID via Dirichlet(alpha)")
    print("  pathological — K classes per client (McMahan et al. 2017)")


if __name__ == "__main__":
    main()
