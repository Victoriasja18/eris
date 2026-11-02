"""
Reads the isoform groups built by scripts/build_isoform_groups.py, which the benchmarks score by (docs/adr/0004).
"""
import csv
from pathlib import Path

ISOFORM_GROUPS = Path(__file__).parent / "isoform_groups.tsv"


def read_isoform_groups(path: Path = ISOFORM_GROUPS) -> dict[str, str]:
    """Returns {IS name: group name} for every name in a group of two or more."""
    with open(path) as handle:
        rows = csv.DictReader((line for line in handle if not line.startswith('#')), delimiter='\t')
        return {r['name']: r['group'] for r in rows}


def isoform_of(name: str, groups: dict[str, str]) -> str:
    """The isoform group of a target or truth IS name, e.g. 'IS15DI_IS6_unknown' or 'IS15DI_2' -> 'IS26'."""
    name = name.split('_')[0]
    return groups.get(name, name)
