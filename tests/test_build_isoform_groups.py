import importlib.util
from pathlib import Path

import pytest

from isoforms import ISOFORM_GROUPS, isoform_of, read_isoform_groups

# The builder is a script, not part of the eris package
_SCRIPT = Path(__file__).parent.parent / "scripts" / "build_isoform_groups.py"
_spec = importlib.util.spec_from_file_location("build_isoform_groups", _SCRIPT)
group_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(group_builder)

# Rows shaped like ISfinder's IS.csv, with its misspelt 'Synomyns' header
TABLE = (
    'N°,Name,Family,Group,Synomyns,Iso,Origin\n'
    '1,IS26,IS6,NA,"IS6, IS140, I",,Proteus vulgaris\n'
    '2,IS15DI,IS6,NA,IS15-DeltaI,IS26,Salmonella\n'
    '3,IS15,IS6,NA,"IS15L, IS1522",IS26,Salmonella\n'
    '4,IS66,IS66,NA,"IS66-1, I",IS66,Agrobacterium\n'
    '5,ISSth1,IS3,NA,,,Streptococcus\n'
    '6,ISSth1b,ISL3,NA,,ISSth1,Streptococcus\n'
    '7,ISEc52,IS3,NA,NA,,Escherichia coli\n'
)


@pytest.fixture
def built(tmp_path):
    table = tmp_path / "IS.csv"
    table.write_text(TABLE)
    return group_builder.build(group_builder.read_table(table))


def test_joins_synonyms_and_isoforms_under_the_reference_is(built):
    groups, _ = built
    assert {n: groups[n] for n in ('IS26', 'IS15', 'IS15DI')} == {'IS26': 'IS26', 'IS15': 'IS26', 'IS15DI': 'IS26'}


def test_tokens_that_are_not_is_names_join_nothing(built):
    # 'I' is a synonym of both IS26 and IS66 in the real table
    groups, _ = built
    assert 'IS66' not in groups and 'I' not in groups and 'IS15L' not in groups


def test_cross_family_joins_are_refused(built):
    groups, refused = built
    assert 'ISSth1' not in groups and refused == [('ISSth1b', 'ISSth1')]


def test_names_alone_are_not_written(built, tmp_path):
    groups, _ = built
    out = tmp_path / "groups.tsv"
    group_builder.write_groups(out, groups)
    assert read_isoform_groups(out) == {'IS15': 'IS26', 'IS15DI': 'IS26', 'IS26': 'IS26'}
    assert isoform_of('ISEc52_IS3_IS3', read_isoform_groups(out)) == 'ISEc52'


# The committed groups -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("names", [
    ('IS26', 'IS15', 'IS15DI', 'IS15DIV'), ('ISEc9', 'ISEcp1'), ('IS600', 'ISSd1'), ('IS1A', 'IS1R'),
    ('IS1203', 'IS629'), ('IS30', 'IS30D'),
])
def test_committed_groups_join_the_names_eris_swaps_on_ec958(names):
    groups = read_isoform_groups()
    assert len({isoform_of(n, groups) for n in names}) == 1


@pytest.mark.parametrize("a, b", [('IS1A', 'IS1F'), ('IS26', 'IS66'), ('ISEc52', 'ISEc53'), ('IS629', 'ISSen1')])
def test_committed_groups_keep_distinct_elements_apart(a, b):
    groups = read_isoform_groups()
    assert isoform_of(a, groups) != isoform_of(b, groups)


@pytest.mark.skipif(not (Path("data") / "IS.mmi").exists(), reason="target database not found in data/")
def test_committed_groups_stay_within_one_family_of_the_target_database():
    import mappy
    # Target names are '<name>_<family>_<group>', e.g. 'IS600_IS3_IS3'
    families = {n.rsplit('_', 2)[0]: n.rsplit('_', 2)[1]
                for n in mappy.Aligner(str(Path("data") / "IS.mmi")).seq_names if n.count('_') >= 2}
    by_group = {}
    for name, group in read_isoform_groups(ISOFORM_GROUPS).items():
        if name in families:
            by_group.setdefault(group, set()).add(families[name])
    assert {g: f for g, f in by_group.items() if len(f) > 1} == {}
