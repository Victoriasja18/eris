from types import SimpleNamespace

from eris.io import ReportRow, format_identities


def test_report_header_puts_target_identity_after_target():
    assert ReportRow.header().split("\t")[:3] == ["locus_id", "target", "target_identity"]
    assert ReportRow.header(mode='collapse').rstrip("\n").split("\t")[-1] == "copy_number"


def test_identities_are_listed_in_target_order_to_four_decimals():
    locus = SimpleNamespace(targets=[SimpleNamespace(identity=0.97314), SimpleNamespace(identity=1.0)])
    assert format_identities(locus) == "0.9731,1.0000"
