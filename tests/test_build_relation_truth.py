import importlib.util
from pathlib import Path

import pytest

# The builder is a script, not part of the eris package
_SCRIPT = Path(__file__).parent.parent / "scripts" / "build_relation_truth.py"
_spec = importlib.util.spec_from_file_location("build_relation_truth", _SCRIPT)
truth_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(truth_builder)

CURATED = (
    '"##Chr=""Chromosome"""\t\t\t\t\t\t\t\t\n'
    '#Chr\tStart\tEnd\tIS_name\tDownstream\tUpstream\tInside\tEffect\t\n'
    # + strand IS; the downstream tag is missing its leading 'E', as in the real table
    'NZ_chr\t1000\t2000\t"ISA;+"\t"C958_RS00002(geneB;+;partial;5bp)"\t"EC958_RS00001 (geneA;-;no partial;100bp)"'
    '\t*\tDisrupts\t\n'
    # - strand IS: downstream is to the left; the 'upstream' gene is curated on the wrong side
    'NZ_chr\t5000\t6000\t"ISB;-"\t"EC958_RS00003(geneC;+;no partial;50bp)"\t"EC958_RS00005(geneE;+;no partial;210bp)"'
    '\t*\tUpregulates\tBring new promoter activity\n'
    # Its upstream gene is ISB's transposase, next door
    'NZ_chr\t6000\t7000\t"ISC;+"\t*\t"EC958_RS00006(tnp;-;no partial;100bp)"\t*\ttranscriptional readthrough?\t\n'
)
GFF = "\n".join("\t".join(f) for f in [
    ("##gff-version 3",),
    ("NZ_chr", "RefSeq", "gene", "801", "900", ".", "-", ".", "locus_tag=EC958_RS00001;gene=geneA"),
    ("NZ_chr", "RefSeq", "CDS", "801", "900", ".", "-", "0", "locus_tag=EC958_RS00001;product=protein A"),
    ("NZ_chr", "RefSeq", "gene", "2006", "2500", ".", "+", ".", "locus_tag=EC958_RS00002;gene=geneB;partial=true"),
    ("NZ_chr", "RefSeq", "gene", "4801", "4950", ".", "+", ".", "locus_tag=EC958_RS00003"),
    ("NZ_chr", "RefSeq", "gene", "4701", "4790", ".", "+", ".", "locus_tag=EC958_RS00005"),
    ("NZ_chr", "RefSeq", "gene", "5101", "5900", ".", "-", ".", "locus_tag=EC958_RS00006"),
]) + "\n"
BED = "chr\t1000\t2000\tISA_1\t0\t+\nchr\t5000\t6000\tISB_1\t0\t-\nchr\t6000\t7000\tISC_1\t0\t+\n"


@pytest.fixture
def built(tmp_path):
    (tmp_path / "curated.txt").write_text(CURATED)
    (tmp_path / "genes.gff").write_text(GFF)
    (tmp_path / "is.bed").write_text(BED)
    truth, review = truth_builder.build(truth_builder.read_curated(tmp_path / "curated.txt"), truth_builder.read_gff(tmp_path / "genes.gff"),
                              truth_builder.read_bed(tmp_path / "is.bed"))
    return {(r["is_id"], r["side"], r["locus_tag"]): r for r in truth}, review


def test_flank_sides_are_read_in_the_is_direction(built):
    truth, _ = built
    assert set(truth) == {
        ("ISA_1", "upstream", "EC958_RS00001"),
        ("ISA_1", "downstream", "EC958_RS00002"),  # tag repaired from 'C958_RS00002'
        ("ISB_1", "inside", "EC958_RS00006"),
        ("ISB_1", "downstream", "EC958_RS00003"),  # left of a - strand IS
        ("ISB_1", "downstream", "EC958_RS00005"),  # curated as upstream; the coordinates win
        ("ISC_1", "upstream", "EC958_RS00006"),
    }
    assert truth[("ISA_1", "downstream", "EC958_RS00002")]["distance_bp"] == 5
    assert truth[("ISB_1", "downstream", "EC958_RS00003")]["distance_bp"] == 50


def test_disrupted_needs_a_partial_flank_of_a_disrupting_is(built):
    truth, _ = built
    assert truth[("ISA_1", "downstream", "EC958_RS00002")]["effect"] == "disrupted"
    assert truth[("ISA_1", "upstream", "EC958_RS00001")]["effect"] == "none"


def test_curated_effect_is_kept_and_upregulates_scores_as_none(built):
    truth, _ = built
    isb = truth[("ISB_1", "downstream", "EC958_RS00003")]
    assert (isb["curated_effect"], isb["effect"], isb["note"]) == ("Upregulates", "none", "Bring new promoter activity")
    assert truth[("ISC_1", "upstream", "EC958_RS00006")]["effect"] == "uncertain"


def test_flank_inside_another_is_is_marked(built):
    truth, _ = built
    assert truth[("ISC_1", "upstream", "EC958_RS00006")]["inside_is"] == "ISB_1"
    assert truth[("ISA_1", "upstream", "EC958_RS00001")]["inside_is"] == ""


def test_review_lists_side_swaps_and_unusual_effects_but_not_upregulates(built):
    _, review = built
    assert sorted((r["is_id"], r["reason"]) for r in review) == [
        ("ISB_1", "curated as upstream, coordinates say downstream"),
        ("ISC_1", 'curated effect is "transcriptional readthrough?"'),
    ]
