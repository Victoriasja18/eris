"""
Builds the isoform groups the benchmarks score by, from ISfinder's synonym and isoform annotation.

ISfinder names some insertion sequences twice: under synonyms (the `Synomyns` column of its IS.csv) and as isoforms
of a reference IS (the `Iso` column), which are near-identical in sequence. minimap2 picks whichever name aligns best,
so the benchmarks treat every name in one isoform group as the same IS (docs/adr/0004).

Each row's name is joined to the names in its Synomyns and Iso cells, transitively, with two guards:
- a cell token is only joined when it is itself an IS name in the table, so junk such as `I` (listed as a synonym of
  both IS26 and IS66) and loose aliases eris never reports (`IS15L`) can't link groups;
- two names of different families are never joined; each refused join is printed.

A group is named after the IS its members' Iso cells point to most, else its alphabetically first member. Only groups
with more than one member are written; any other name is its own group.

Usage:
    curl -sLo IS.csv https://raw.githubusercontent.com/thanhleviet/ISfinder-sequences/bedbeb5/IS.csv
    uv run python scripts/build_isoform_groups.py IS.csv [-o tests/isoform_groups.tsv]
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SOURCE = "https://github.com/thanhleviet/ISfinder-sequences (IS.csv at bedbeb5, 2020-10-06)"
DEFAULT_OUT = Path(__file__).parent.parent / "tests" / "isoform_groups.tsv"


def read_table(path: Path) -> list[dict]:
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def tokens(cell: str) -> list[str]:
    return [t for t in (t.strip() for t in cell.split(',')) if t and t.upper() != 'NA']


def build(rows: list[dict]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Returns ({name: group} for names in groups of two or more, [refused cross-family joins])."""
    family = {r['Name'].strip(): r['Family'].strip() for r in rows}
    parent = {name: name for name in family}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    refused, iso_votes = [], Counter()
    for r in rows:
        name = r['Name'].strip()
        for col in ('Synomyns', 'Iso'):
            for other in tokens(r[col]):
                if other not in family or other == name:
                    continue
                if family[other] != family[name]:
                    refused.append((name, other))
                    continue
                parent[find(name)] = find(other)
                if col == 'Iso':
                    iso_votes[other] += 1

    members = {}
    for name in family:
        members.setdefault(find(name), []).append(name)
    groups = {}
    for names in members.values():
        if len(names) < 2:
            continue
        label = min(names, key=lambda n: (-iso_votes[n], n))
        groups.update((n, label) for n in names)
    return groups, refused


def write_groups(path: Path, groups: dict[str, str]):
    with open(path, 'w') as out:
        out.write(f"# Isoform groups from {SOURCE}; built by scripts/build_isoform_groups.py\n")
        out.write("# A name not listed here is its own group.\n")
        out.write("name\tgroup\n")
        for name, group in sorted(groups.items(), key=lambda item: (item[1], item[0])):
            out.write(f"{name}\t{group}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('table', type=Path, help="ISfinder's IS.csv")
    parser.add_argument('-o', '--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    groups, refused = build(read_table(args.table))
    for name, other in refused:
        print(f"refused cross-family join: {name} - {other}", file=sys.stderr)
    write_groups(args.out, groups)
    print(f"{len(set(groups.values()))} groups over {len(groups)} names -> {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
