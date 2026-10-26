import argparse

import pytest

from eris.cli import fraction


@pytest.mark.parametrize("value, expected", [("0", 0.0), ("0.9", 0.9), ("1", 1.0)])
def test_fraction_accepts_values_from_zero_to_one(value, expected):
    assert fraction(value) == expected


@pytest.mark.parametrize("value", ["90", "-0.1", "1.01", "high"])
def test_fraction_rejects_values_outside_zero_to_one(value):
    with pytest.raises(argparse.ArgumentTypeError):
        fraction(value)
