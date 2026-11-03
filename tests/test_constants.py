import pytest
from eris.constants import Strand, Context, FeatureType, Effect, Orientation

def test_strand_enum():
    assert Strand.FORWARD == 1
    assert Strand.REVERSE == -1
    assert Strand.UNSTRANDED == 0

    # Coercion tests
    assert Strand('+') == Strand.FORWARD
    assert Strand('1') == Strand.FORWARD
    assert Strand('+1') == Strand.FORWARD
    assert Strand(b'+') == Strand.FORWARD
    assert Strand('-') == Strand.REVERSE
    assert Strand('-1') == Strand.REVERSE
    assert Strand(b'-') == Strand.REVERSE
    assert Strand('.') == Strand.UNSTRANDED
    assert Strand('unknown') == Strand.UNSTRANDED

    # String representation
    assert str(Strand.FORWARD) == '+'
    assert str(Strand.REVERSE) == '-'
    assert str(Strand.UNSTRANDED) == '.'

def test_context_enum():
    assert Context.UPSTREAM != Context.DOWNSTREAM
    assert Context.INSIDE != Context.OVERLAPPING

def test_feature_type():
    assert FeatureType.CDS == "CDS"
    assert FeatureType.MOBILE_ELEMENT == "mobile_element"

def test_effect_flags():
    e = Effect.NONE
    assert e == Effect.NONE
    e_combined = Effect.TRUNCATED | Effect.DISRUPTED
    assert Effect.TRUNCATED in e_combined
    assert Effect.DISRUPTED in e_combined

def test_orientation_enum():
    assert Orientation.SAME == "same strand"
    assert Orientation.OPPOSITE == "opposite strand"
    assert Orientation.NONE == "-"
