#!/usr/bin/env python3
"""
Benchmark minimap2 parameters for the eris IS element detection pipeline.

Inputs
------
genome      : complete reference genome FASTA
target_db   : IS element database FASTA (the -t input to eris)
ground_truth: FASTA whose headers are ">IS_name chrom:start-end"

Optional
--------
--mmi-baseline: pre-built .mmi index to include as a baseline row.
  Index-time params (preset, k, w) are baked into the .mmi and cannot be
  changed; only search-time params (best-n, min-cnt, min-chain-score, bw)
  are varied against it.

The script patches TargetDatabase._aligner at runtime — no pipeline code is
changed.  Predicted loci are compared to ground truth by:
  - same IS target name (exact match)
  - positional overlap, with boundaries within --tolerance bp

Usage
-----
# Single run with FASTA target
python benchmark_minimap2.py genome.fasta targets.fasta ground_truth.fasta \
    --preset asm10 --best-n 20

# Run using pre-built MMI only
python benchmark_minimap2.py genome.fasta targets.mmi ground_truth.fasta

# Grid sweep: compare pre-built MMI baseline against FASTA param combos
python benchmark_minimap2.py genome.fasta targets.fasta ground_truth.fasta \
    --mmi-baseline targets.mmi --sweep
"""
import argparse
import re
import sys
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import mappy

# --------------------------------------------------------------------------- #
# Ground-truth parsing
# --------------------------------------------------------------------------- #
_GT_RE = re.compile(r"^(?P<name>\S+)\s+(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)")


@dataclass(frozen=True)
class TruthEntry:
    name: str       # IS element name
    chrom: str
    start: int      # 0-based
    end: int        # exclusive


def parse_ground_truth(fasta_path: Path) -> list[TruthEntry]:
    entries = []
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            m = _GT_RE.match(header)
            if not m:
                raise ValueError(
                    f"Cannot parse ground-truth header: {header!r}\n"
                    "Expected format: >IS_name chrom:start-end"
                )
            entries.append(TruthEntry(
                name=m.group("name"),
                chrom=m.group("chrom"),
                start=int(m.group("start")),
                end=int(m.group("end")),
            ))
    return entries


# --------------------------------------------------------------------------- #
# Pipeline runner with patched aligner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PredictedLocus:
    target: str     # IS element name (first target on the locus)
    contig: str
    start: int
    end: int


def run_pipeline(genome_path: Path, target_db_path: Path, aligner_kwargs: dict) -> list[PredictedLocus]:
    """Run the eris LinearPipeline with a custom mappy.Aligner."""
    from eris.io import TargetDatabase, GenomeAssembly
    from eris.pipeline import LinearPipeline
    from eris.constants import FeatureType

    db = TargetDatabase(target_db_path, feature_type=FeatureType.IS)

    # Build a custom aligner with the test params and inject it
    custom_aligner = mappy.Aligner(fn_idx_in=str(target_db_path), **aligner_kwargs)
    if not custom_aligner:
        raise ValueError(f"mappy failed to load: {target_db_path}")
    db._aligner = custom_aligner

    genome = GenomeAssembly.from_file(genome_path)
    pipeline = LinearPipeline(target_db=db)

    results = []
    for locus in pipeline(genome):
        if not locus.targets:
            continue
        target_name = locus.targets[0].id.split("_")[0] if "_" in locus.targets[0].id else locus.targets[0].id
        # Use the first segment's contig/coords for multi-contig loci
        contig = locus.contig.split("|")[0]
        results.append(PredictedLocus(
            target=target_name,
            contig=contig,
            start=locus.start,
            end=locus.end,
        ))
    return results


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __str__(self) -> str:
        return (f"TP={self.tp}  FP={self.fp}  FN={self.fn}  "
                f"P={self.precision:.3f}  R={self.recall:.3f}  F1={self.f1:.3f}")


def overlaps(p: PredictedLocus, t: TruthEntry, tolerance: int) -> bool:
    """True if same IS name, contigs match, intervals overlap, and boundaries within tolerance."""
    if p.target != t.name:
        return False
    if p.contig != t.chrom:
        return False
    # Positional overlap
    if p.end <= t.start or t.end <= p.start:
        return False
    # Boundary tolerance
    if abs(p.start - t.start) > tolerance:
        return False
    if abs(p.end - t.end) > tolerance:
        return False
    return True


def evaluate(predicted: list[PredictedLocus], truth: list[TruthEntry], tolerance: int) -> Metrics:
    m = Metrics()
    matched_truth = set()

    for pred in predicted:
        matched = False
        for i, t in enumerate(truth):
            if i not in matched_truth and overlaps(pred, t, tolerance):
                matched_truth.add(i)
                matched = True
                break
        if matched:
            m.tp += 1
        else:
            m.fp += 1

    m.fn = len(truth) - len(matched_truth)
    return m


# --------------------------------------------------------------------------- #
# Parameter grid
# --------------------------------------------------------------------------- #
# Index-time params: baked into the .mmi; ignored when loading a pre-built index.
# Search-time params: applied at query time; work with both FASTA and .mmi.
SWEEP_GRID_FASTA = {
    "preset":          ["map-ont", "asm5", "asm10", "asm20"],
    "best_n":          [5, 10, 20, 50],
    "min_cnt":         [2, 3, 5],
    "min_chain_score": [20, 40, 80],
}

SWEEP_GRID_MMI = {           # only search-time params apply to a pre-built index
    "best_n":          [5, 10, 20, 50],
    "min_cnt":         [2, 3, 5],
    "min_chain_score": [20, 40, 80],
}


def build_aligner_kwargs(args) -> dict:
    kwargs = {}
    if args.preset:
        kwargs["preset"] = args.preset
    if args.best_n is not None:
        kwargs["best_n"] = args.best_n
    if args.min_cnt is not None:
        kwargs["min_cnt"] = args.min_cnt
    if args.min_chain_score is not None:
        kwargs["min_chain_score"] = args.min_chain_score
    if args.bw is not None:
        kwargs["bw"] = args.bw
    if args.k is not None:
        kwargs["k"] = args.k
    if args.w is not None:
        kwargs["w"] = args.w
    return kwargs


def label(kwargs: dict, source: str = "") -> str:
    parts = [f"[{source}]"] if source else []
    parts += [f"{k}={v}" for k, v in kwargs.items()]
    return "  ".join(parts) if parts else "(defaults)"


def _sweep_rows(genome: Path, db_path: Path, grid: dict, source_label: str,
                truth: list, tolerance: int) -> tuple[float, dict]:
    """Run one sweep grid, print TSV rows, return (best_f1, best_kwargs)."""
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"# {source_label}: {len(combos)} combinations", file=sys.stderr)

    best_f1, best_kwargs = -1.0, {}
    for combo in combos:
        kwargs = dict(zip(keys, combo))
        try:
            predicted = run_pipeline(genome, db_path, kwargs)
            m = evaluate(predicted, truth, tolerance)
        except Exception as e:
            print(f"{source_label}\t" + "\t".join(str(v) for v in combo) + f"\tERROR\tERROR\tERROR\tERROR\tERROR\t{e}")
            continue

        row_vals = [source_label] + [str(v) for v in combo] + [
            str(m.tp), str(m.fp), str(m.fn),
            f"{m.precision:.3f}", f"{m.recall:.3f}", f"{m.f1:.3f}",
        ]
        print("\t".join(row_vals))

        if m.f1 > best_f1:
            best_f1 = m.f1
            best_kwargs = kwargs

    return best_f1, best_kwargs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Benchmark minimap2 params for eris IS detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("genome",       type=Path, help="Complete reference genome FASTA")
    p.add_argument("target_db",    type=Path, help="IS database FASTA or pre-built .mmi")
    p.add_argument("ground_truth", type=Path, help="Ground-truth FASTA (>IS_name chrom:start-end)")

    p.add_argument("--mmi-baseline", type=Path, default=None, dest="mmi_baseline", metavar="FILE",
                   help="Pre-built .mmi index to include as a baseline row in --sweep. "
                        "Index-time params (preset, k, w) are baked in and cannot be varied; "
                        "only search-time params (best-n, min-cnt, min-chain-score, bw) are swept.")

    mm2 = p.add_argument_group("minimap2 params (single run; ignored in --sweep)")
    mm2.add_argument("--preset",          default=None,
                     help="Index-time preset, e.g. asm10, asm20, map-ont. "
                          "Ignored when target_db is a .mmi.")
    mm2.add_argument("--best-n",          type=int, default=None, dest="best_n")
    mm2.add_argument("--min-cnt",         type=int, default=None, dest="min_cnt")
    mm2.add_argument("--min-chain-score", type=int, default=None, dest="min_chain_score")
    mm2.add_argument("--bw",              type=int, default=None,
                     help="Bandwidth (search-time). Works with both FASTA and .mmi.")
    mm2.add_argument("--k",               type=int, default=None,
                     help="k-mer size (index-time). Ignored when target_db is a .mmi.")
    mm2.add_argument("--w",               type=int, default=None,
                     help="Minimizer window (index-time). Ignored when target_db is a .mmi.")

    p.add_argument("--tolerance", type=int, default=75,
                   help="Max bp difference between predicted and true boundaries")
    p.add_argument("--sweep", action="store_true",
                   help="Run grid sweep instead of a single run")

    args = p.parse_args()

    is_mmi = args.target_db.suffix == ".mmi"

    truth = parse_ground_truth(args.ground_truth)
    print(f"Ground truth: {len(truth)} entries", file=sys.stderr)

    if args.sweep:
        all_keys = sorted(
            set(SWEEP_GRID_FASTA.keys()) | set(SWEEP_GRID_MMI.keys()) | {"source"}
        )
        stat_keys = ["TP", "FP", "FN", "P", "R", "F1"]
        print("\t".join(["source"] + list(SWEEP_GRID_FASTA.keys()) + stat_keys))

        best_f1_overall, best_label = -1.0, ""

        # --- MMI baseline sweep (search-time params only) ---
        if args.mmi_baseline:
            f1, kwargs = _sweep_rows(
                args.genome, args.mmi_baseline, SWEEP_GRID_MMI,
                "mmi_baseline", truth, args.tolerance,
            )
            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_label = label(kwargs, "mmi_baseline")

        # --- FASTA sweep (all params) ---
        if not is_mmi:
            f1, kwargs = _sweep_rows(
                args.genome, args.target_db, SWEEP_GRID_FASTA,
                args.target_db.stem, truth, args.tolerance,
            )
            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_label = label(kwargs, args.target_db.stem)
        elif not args.mmi_baseline:
            # target_db is itself a .mmi, no FASTA provided
            f1, kwargs = _sweep_rows(
                args.genome, args.target_db, SWEEP_GRID_MMI,
                args.target_db.stem, truth, args.tolerance,
            )
            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_label = label(kwargs, args.target_db.stem)

        print(f"\nBest overall F1={best_f1_overall:.3f}: {best_label}", file=sys.stderr)

    else:
        kwargs = build_aligner_kwargs(args)
        if is_mmi and any(k in kwargs for k in ("preset", "k", "w")):
            print("Note: preset/k/w are ignored for pre-built .mmi files", file=sys.stderr)
        print(f"Running with: {label(kwargs)}", file=sys.stderr)
        predicted = run_pipeline(args.genome, args.target_db, kwargs)
        print(f"Predicted loci: {len(predicted)}", file=sys.stderr)
        m = evaluate(predicted, truth, args.tolerance)
        print(m)


if __name__ == "__main__":
    main()
