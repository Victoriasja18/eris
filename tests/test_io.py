from types import SimpleNamespace

from eris.constants import Strand
from eris.io import OutputManager, ReportRow, format_contig_path, format_identities
from eris.pipeline import GenomicFeature, LocationSegment, Locus


def test_report_header_puts_target_identity_after_target():
    assert ReportRow.header().split("\t")[:3] == ["locus_id", "target", "target_identity"]
    assert ReportRow.header(mode='collapse').rstrip("\n").split("\t")[-1] == "copy_number"


def test_report_header_puts_contig_path_after_target_identity_in_both_modes():
    for mode in ('variant', 'collapse'):
        assert ReportRow.header(mode).split("\t")[2:4] == ["target_identity", "contig_path"]


def test_identities_are_listed_in_target_order_to_four_decimals():
    locus = SimpleNamespace(targets=[SimpleNamespace(identity=0.97314), SimpleNamespace(identity=1.0)])
    assert format_identities(locus) == "0.9731,1.0000"


def _locus(id_, *segments):
    target = GenomicFeature("IS1", None, [LocationSegment(*s) for s in segments], identity=1.0)
    contig = "|".join(s[0] for s in segments)
    return Locus(id_, contig, segments[0][1], segments[-1][2], [target], [], [], [])


def test_contig_path_lists_each_contig_with_the_strand_it_is_read_on():
    stitched = _locus("locus_split_1", ("ctgA", 900, 1000, Strand.FORWARD), ("ctgB", 0, 500, Strand.REVERSE),
                      ("ctgC", 0, 500, Strand.FORWARD))
    assert format_contig_path(stitched) == "ctgA+,ctgB-,ctgC+"


def test_contig_path_of_a_local_locus_is_its_contig_on_the_target_strand():
    assert format_contig_path(_locus("locus_1", ("ctg5", 100, 1100, Strand.REVERSE))) == "ctg5-"


def test_cds_fasta_names_each_gene_by_the_report_gene_id(tmp_path):
    # pyfgs coordinates are 0-based and half-open; a frameshift-corrected call is longer than end - start
    genes = [SimpleNamespace(start=10, end=19, strand=1, sequence=lambda: b"ATGAAATAA"),
             SimpleNamespace(start=30, end=39, strand=-1, sequence=lambda: b"ATGAAACTAA")]
    prefix = str(tmp_path / "out")
    with OutputManager(prefix, write_gff=False, write_faa=False, write_fna=True) as out:
        out.write_global_genes("ctg_1", "", genes)

    assert (tmp_path / "out_cds.fna").read_text() == (">ctg_1_10_19 strand=+\nATGAAATAA\n"
                                                      ">ctg_1_30_39 strand=-\nATGAAACTAA\n")


def test_report_and_locus_fasta_carry_the_contig_path(tmp_path):
    locus = _locus("locus_split_1", ("ctgA", 900, 1000, Strand.FORWARD), ("ctgB", 0, 500, Strand.REVERSE))
    prefix = str(tmp_path / "out")
    with OutputManager(prefix, write_gff=False, write_faa=False, write_fna=False, mode='collapse') as out:
        out.write_locus_relations(locus)
        out.write_locus_fasta(locus, "ACGT")

    header, row = (tmp_path / "out_report.tsv").read_text().splitlines()
    assert dict(zip(header.split("\t"), row.split("\t")))["contig_path"] == "ctgA+,ctgB-"
    assert (tmp_path / "out_loci.fasta").read_text() == ">locus_split_1 path=ctgA+,ctgB-\nACGT\n"
