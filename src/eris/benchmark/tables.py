"""
Tables of a relation score: long-format metrics, confusion matrices, and one row per truth relation or false flank.

Two counts sit behind each side's numbers. Recall counts truth relations found on their side (`tp` of `fn`).
Precision counts eris's first flanks per side per branch (`called`) that match a truth relation on that side
(`called_true`): a collapsed repeat can report one truth relation from several branches, so `called_true` is not `tp`.
A flank found on the wrong side counts against both sides.
"""
import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import median

from eris.benchmark.relations import BANDS, DISRUPTING, SIDES, FlankRow, RelationScore, TruthRow, band

ALL = 'all'
MATRIX_SIDES = (*SIDES, 'not_reported')
TRUTH_EFFECTS = ('disrupted', 'none', 'unscored')
RELATION_COLUMNS = (
    'status', 'is_group', 'side', 'locus_tag', 'truth_is', 'replicon', 'truth_start', 'truth_end', 'truth_strand',
    'truth_distance', 'truth_band', 'truth_effect', 'eris_locus', 'eris_gene', 'eris_side', 'eris_hops',
    'eris_distance', 'eris_band', 'eris_effect', 'sequence_ok', 'miss_cause')


def _ratio(n: int, d: int) -> float | str:
    return n / d if d else ''


def _f1(precision: float | str, recall: float | str) -> float | str:
    if precision == '' or recall == '' or precision + recall == 0:
        return ''
    return 2 * precision * recall / (precision + recall)


def truth_label(row: TruthRow) -> str:
    return row.truth.effect or 'unscored'


def metrics(score: RelationScore) -> list[dict]:
    """
    Long-format metrics: one row per (dimension, side, replicon, metric). Dimensions are 'relation' (per side and
    'both'), 'disrupted' and 'distance' (on relations found on their side). Replicon 'all' includes flanks that could
    not be placed on the reference.
    """
    replicons = sorted({r.truth.replicon for r in score.truth_rows} | {r.replicon for r in score.flank_rows} - {''})
    rows = []
    for replicon in (ALL, *replicons):
        truths = [r for r in score.truth_rows if replicon == ALL or r.truth.replicon == replicon]
        flanks = [r for r in score.flank_rows if replicon == ALL or r.replicon == replicon]
        for side in (*SIDES, 'both'):
            rows += _side_metrics(replicon, side, [r for r in truths if side in ('both', r.truth.side)],
                                  [r for r in flanks if side in ('both', r.flank.side)])
        rows += _disrupted_metrics(replicon, truths, flanks)
        for side in (*SIDES, 'both'):
            rows += _distance_metrics(replicon, side, [r for r in truths if side in ('both', r.truth.side)])
    return rows


def _rows(dimension: str, side: str, replicon: str, values: dict) -> list[dict]:
    return [{'dimension': dimension, 'side': side, 'replicon': replicon, 'metric': k, 'value': v}
            for k, v in values.items()]


def _side_metrics(replicon: str, side: str, truths: list[TruthRow], flanks: list[FlankRow]) -> list[dict]:
    tp = sum(r.status == 'found' for r in truths)
    called_true = sum(r.status == 'true' for r in flanks)
    precision, recall = _ratio(called_true, len(flanks)), _ratio(tp, len(truths))
    values = {'truth': len(truths), 'tp': tp, 'fn': len(truths) - tp,
              'fn_wrong_side': sum(r.status == 'wrong_side' for r in truths),
              'called': len(flanks), 'called_true': called_true, 'fp': len(flanks) - called_true,
              'precision': precision, 'recall': recall, 'f1': _f1(precision, recall)}
    if side == 'both':
        values['pooled_recall'] = _ratio(sum(r.status != 'missed' for r in truths), len(truths))
    return _rows('relation', side, replicon, values)


def _disrupted_metrics(replicon: str, truths: list[TruthRow], flanks: list[FlankRow]) -> list[dict]:
    disrupted = [r for r in truths if r.truth.effect == 'disrupted']
    tp = sum(r.status == 'found' and bool(r.flank.effect & DISRUPTING) for r in disrupted)
    called = [r for r in flanks if r.flank.effect & DISRUPTING and r.truth_effect != 'unscored']
    called_true = sum(r.truth_effect == 'disrupted' for r in called)
    precision, recall = _ratio(called_true, len(called)), _ratio(tp, len(disrupted))
    return _rows('disrupted', 'both', replicon, {
        'truth': len(disrupted), 'tp': tp, 'fn': len(disrupted) - tp, 'called': len(called),
        'called_true': called_true, 'fp': len(called) - called_true,
        'precision': precision, 'recall': recall, 'f1': _f1(precision, recall)})


def _distance_metrics(replicon: str, side: str, truths: list[TruthRow]) -> list[dict]:
    found = [r for r in truths if r.status == 'found']
    # eris reports an overlap as distance 0
    errors = [abs(r.flank.distance_bp - max(0, r.truth.distance_bp)) for r in found]
    agree = sum(band(r.truth.distance_bp) == band(r.flank.distance_bp, r.flank.overlapping) for r in found)
    return _rows('distance', side, replicon, {
        'found': len(found), 'band_agreement': _ratio(agree, len(found)),
        'median_abs_error': median(errors) if errors else '', 'max_abs_error': max(errors, default=''),
        'within_10bp': _ratio(sum(e <= 10 for e in errors), len(errors)),
        'within_50bp': _ratio(sum(e <= 50 for e in errors), len(errors))})


def side_confusion(score: RelationScore) -> tuple[tuple[str, ...], dict[str, Counter]]:
    """Truth side x the side eris reported it on; the 'no_truth' row counts first flanks that match no truth."""
    matrix = {side: Counter() for side in (*SIDES, 'no_truth')}
    for r in score.truth_rows:
        matrix[r.truth.side][r.flank.side if r.flank else 'not_reported'] += 1
    for r in score.flank_rows:
        if r.status == 'no_truth':
            matrix['no_truth'][r.flank.side] += 1
    return MATRIX_SIDES, matrix


def distance_confusion(score: RelationScore) -> tuple[tuple[str, ...], dict[str, Counter]]:
    """Truth distance band x eris's, on relations found on their side."""
    matrix = {b: Counter() for b in BANDS}
    for r in score.truth_rows:
        if r.status == 'found':
            matrix[band(r.truth.distance_bp)][band(r.flank.distance_bp, r.flank.overlapping)] += 1
    return BANDS, matrix


def effect_confusion(score: RelationScore) -> tuple[tuple[str, ...], dict[str, Counter]]:
    """Truth effect x eris's effect, on relations found on their side; 'unscored' rows are shown but never scored."""
    matrix = {e: Counter() for e in TRUTH_EFFECTS}
    for r in score.truth_rows:
        if r.status == 'found':
            matrix[truth_label(r)][r.flank.effect.name] += 1
    columns = ['NONE', 'DISRUPTED', 'TRUNCATED', 'UPREGULATED']
    columns += sorted({c for counts in matrix.values() for c in counts} - set(columns))
    return tuple(columns), matrix


def relation_rows(score: RelationScore) -> list[dict]:
    """
    One row per truth relation (TP, FN, or FN_wrong_side when eris reported the gene on the other side only) and per
    first flank that matches no truth relation on its side (FP, or FP_wrong_side).
    """
    rows = []
    status = {'found': 'TP', 'missed': 'FN', 'wrong_side': 'FN_wrong_side'}
    for r in score.truth_rows:
        t, f = r.truth, r.flank
        rows.append({
            'status': status[r.status], 'is_group': t.group, 'side': t.side, 'locus_tag': t.locus_tag,
            'truth_is': t.is_id, 'replicon': t.replicon, 'truth_start': t.start, 'truth_end': t.end,
            'truth_strand': t.strand, 'truth_distance': t.distance_bp, 'truth_band': band(t.distance_bp),
            'truth_effect': truth_label(r), **_flank_columns(f),
            'sequence_ok': '' if r.sequence_ok is None else r.sequence_ok, 'miss_cause': r.miss_cause})
    for r in score.flank_rows:
        if r.status != 'true':
            rows.append({'status': 'FP' if r.status == 'no_truth' else 'FP_wrong_side', 'is_group': r.flank.group,
                         'side': r.flank.side, 'replicon': r.replicon, **_flank_columns(r.flank)})
    return rows


def _flank_columns(f) -> dict:
    if f is None:
        return {}
    return {'eris_locus': f.locus_id, 'eris_gene': f.gene_id, 'eris_side': f.side, 'eris_hops': f.hops,
            'eris_distance': f.distance_bp, 'eris_band': band(f.distance_bp, f.overlapping), 'eris_effect': f.effect.name}


def _format(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_tsv(path: Path, columns: Iterable[str], rows: Iterable[dict]):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter='\t', restval='', lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: _format(v) for k, v in row.items()} for row in rows)


def write_matrix(path: Path, matrix: tuple[tuple[str, ...], dict[str, Counter]]):
    columns, counts = matrix
    write_tsv(path, ('truth', *columns), ({'truth': row, **{c: n[c] for c in columns}} for row, n in counts.items()))


def write_tables(score: RelationScore, outdir: Path):
    """Writes metrics.tsv, relations.tsv and the side, distance and effect confusion matrices."""
    outdir.mkdir(parents=True, exist_ok=True)
    write_tsv(outdir / 'metrics.tsv', ('dimension', 'side', 'replicon', 'metric', 'value'), metrics(score))
    write_tsv(outdir / 'relations.tsv', RELATION_COLUMNS, relation_rows(score))
    write_matrix(outdir / 'side_confusion.tsv', side_confusion(score))
    write_matrix(outdir / 'distance_confusion.tsv', distance_confusion(score))
    write_matrix(outdir / 'effect_confusion.tsv', effect_confusion(score))
