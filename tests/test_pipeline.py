import pytest
import numpy as np
from Bio.Seq import Seq

from eris.alignment import AlignmentBatch
from eris.constants import FeatureType, Strand
from eris.graph import Edge, TopologyEngine
from eris.io import GenomeAssembly, format_contig_path
from eris.pipeline import LocusBuilder, Locus, GenomicFeature, LocationSegment


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


def test_stitched_locus_reports_the_strand_each_contig_is_read_on():
    # IS1 runs off ctgA, through the unaligned ctgB read backwards ('ctgA+ -> ctgB-'), and into ctgC
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=0),
             Edge("ctgB", Strand.REVERSE, "ctgC", Strand.FORWARD, overlap=0)]
    builder = _builder({"ctgA": 1000, "ctgB": 500, "ctgC": 1000}, edges)
    alignments = {
        "ctgA": _batch([("IS1", 2000, 0, 1000, "ctgA", 1000, 0, 1000, 1000, 1000)]),
        "ctgC": _batch([("IS1", 2000, 1500, 2000, "ctgC", 1000, 0, 500, 500, 500)]),
    }

    loci = list(builder.assemble(alignments, min_coverage=0.30, min_identity=0.0))

    assert [format_contig_path(locus) for locus in loci] == ["ctgA+,ctgB-,ctgC+"]


def _path_locus(*segments):
    """A locus whose one target runs along segments of (contig, start, end, strand), in the target's direction."""
    target = GenomicFeature("IS1", FeatureType.CDS, [LocationSegment(*s) for s in segments])
    contig = "|".join(s[0] for s in segments)
    return Locus("locus_1", contig, segments[0][1], segments[-1][2], [target], [], [], [])


def _genome(contigs, edges):
    return GenomeAssembly("test", {c: Seq(s) for c, s in contigs.items()}, edges, {}, {c: len(s) for c, s in contigs.items()})


# ctgB read backwards (TTTACGTGGG reversed: GGGTTTACGT) starts with the last 3 bp of ctgA, the edge overlap
_A, _B = "AAAACCCGGG", "ACGTAAACCC"


def test_stitched_sequence_joins_contigs_on_their_path_strands_counting_each_overlap_once():
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgA", 0, 10, Strand.FORWARD), ("ctgB", 0, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "AAAACCCGGG" + "TTTACGT"


def test_stitched_sequence_is_cut_to_the_target_ends_on_the_first_and_last_contigs():
    # The target starts 4 bp into ctgA and ends 7 bp into ctgB as it is read (ctgB 3..10 on its forward strand)
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgA", 4, 10, Strand.FORWARD), ("ctgB", 3, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "CCCGGG" + "TTTA"


def test_stitched_sequence_finds_the_overlap_of_a_link_written_on_the_other_strands():
    # 'ctgB+ -> ctgA-' is the same link as 'ctgA+ -> ctgB-'
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgB", Strand.FORWARD, "ctgA", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgA", 0, 10, Strand.FORWARD), ("ctgB", 0, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "AAAACCCGGG" + "TTTACGT"


def test_stitched_sequence_read_backwards_starts_from_the_target_start_on_the_reverse_strand():
    # The same target walked the other way, 'ctgB+ -> ctgA-'
    genome = _genome({"ctgA": _A, "ctgB": _B}, [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=3)])
    locus = _path_locus(("ctgB", 3, 10, Strand.FORWARD), ("ctgA", 4, 10, Strand.REVERSE))

    assert locus.extract_sequence(genome) == str(Seq("CCCGGG" + "TTTA").reverse_complement())


def test_stitched_sequence_keeps_an_unaligned_intermediate_contig_whole_but_for_its_overlap():
    # 'ctgA+ -> ctgB- -> ctgC+', each link overlapping by 2 bp; the target claims all of the unaligned ctgB
    edges = [Edge("ctgA", Strand.FORWARD, "ctgB", Strand.REVERSE, overlap=2),
             Edge("ctgB", Strand.REVERSE, "ctgC", Strand.FORWARD, overlap=2)]
    genome = _genome({"ctgA": "GGGGAC", "ctgB": "TTTGTC", "ctgC": "AAACCCC"}, edges)
    locus = _path_locus(("ctgA", 3, 6, Strand.FORWARD), ("ctgB", 0, 6, Strand.REVERSE), ("ctgC", 0, 4, Strand.FORWARD))

    assert locus.extract_sequence(genome) == "GAC" + "CAAA" + "AC"


def test_local_sequence_is_its_slice_of_the_contig():
    genome = _genome({"ctgA": _A}, [])
    locus = _path_locus(("ctgA", 2, 8, Strand.REVERSE))

    assert locus.extract_sequence(genome) == "AACCCG"
