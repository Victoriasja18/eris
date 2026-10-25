import pytest
import numpy as np
from Bio.Seq import Seq

from eris.alignment import AlignmentBatch
from eris.constants import Strand
from eris.graph import Edge, TopologyEngine
from eris.io import GenomeAssembly
from eris.pipeline import LocusBuilder


def _batch(rows):
    """rows: (q_name, q_length, q_start, q_end, t_name, t_length, t_start, t_end, length, match)"""
    qn, ql, qs, qe, tn, tl, ts, te, bl, ml = zip(*rows)
    return AlignmentBatch(
        q_names=np.array(qn, dtype=object), q_lengths=np.array(ql, dtype=np.int32),
        q_starts=np.array(qs, dtype=np.int32), q_ends=np.array(qe, dtype=np.int32),
        t_names=np.array(tn, dtype=object), t_lengths=np.array(tl, dtype=np.int32),
        t_starts=np.array(ts, dtype=np.int32), t_ends=np.array(te, dtype=np.int32),
        strands=np.array([Strand.FORWARD] * len(rows), dtype=np.int8),
        lengths=np.array(bl, dtype=np.int32), matches=np.array(ml, dtype=np.int32),
        mismatches=np.array([b - m for b, m in zip(bl, ml)], dtype=np.int32),
        qualities=np.array([60] * len(rows), dtype=np.int32),
        cigars=np.array([f"{b}M" for b in bl], dtype=object)
    )


def _builder(lengths, edges=()):
    depths = {c: 10.0 for c in lengths}
    genome = GenomeAssembly("test", {c: Seq("A" * n) for c, n in lengths.items()}, list(edges), depths, lengths)
    engine = TopologyEngine(list(edges), lengths, depths, mode='variant')
    return LocusBuilder(engine, genome, max_feature_hops=0)


def test_assemble_drops_local_alignments_below_min_identity():
    builder = _builder({"ctgA": 10000})
    alignments = {"ctgA": _batch([
        ("IS1", 1000, 0, 1000, "ctgA", 10000, 1000, 2000, 1000, 990),
        ("IS1", 1000, 0, 1000, "ctgA", 10000, 6000, 7000, 1000, 800),
    ])}

    loci = list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.90))

    assert [(locus.start, [t.identity for t in locus.targets]) for locus in loci] == [(1000, [pytest.approx(0.99)])]


def test_assemble_pools_identity_of_a_stitched_locus_over_its_real_fragments():
    # IS1 runs off the end of ctgA, across the unaligned ctgB, and into the start of ctgC
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0),
             Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0)]
    builder = _builder({"ctgA": 1000, "ctgB": 500, "ctgC": 1000}, edges)
    alignments = {
        "ctgA": _batch([("IS1", 2000, 0, 1000, "ctgA", 1000, 0, 1000, 1000, 950)]),
        "ctgC": _batch([("IS1", 2000, 1500, 2000, "ctgC", 1000, 0, 500, 500, 400)]),
    }

    # (950 + 400) / (1000 + 500) = 0.90; counting ctgB's 500 bp would give 0.675
    kept = list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.85))
    assert [t.identity for locus in kept for t in locus.targets] == [pytest.approx(0.90)]

    assert list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.95)) == []
