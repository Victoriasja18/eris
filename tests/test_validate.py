import csv
import random

import pytest
from Bio.Seq import reverse_complement

from eris.benchmark.cli import main
from eris.benchmark.relations import (
    Flank,
    Placement,
    TruthRelation,
    read_cds_sequences,
    read_report_flanks,
    score_flanks,
)
from eris.benchmark.tables import metrics, relation_rows, side_confusion
from eris.constants import Effect

REPORT_COLUMNS = ('locus_id', 'target', 'target_identity', 'contig_path', 'gene_id', 'context', 'dist_bp', 'topo_hops',
                  'orientation', 'effect', 'fractional_depth')
TRUTH_COLUMNS = ('is_id', 'replicon', 'is_start', 'is_end', 'is_strand', 'side', 'locus_tag', 'gene', 'product',
                 'cds_start', 'cds_end', 'cds_strand', 'partial', 'pseudo', 'distance_bp', 'effect', 'inside_is', 'note')


def write_tsv(path, columns, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t', restval='', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def report_row(locus, target, gene_id, context, dist=0, hops=0, effect='NONE'):
    return {'locus_id': locus, 'target': target, 'gene_id': gene_id, 'context': context, 'dist_bp': dist,
            'topo_hops': hops, 'effect': effect}


def truth_row(is_id, side, tag, start, end, strand='+', distance=10, effect='none'):
    return {'is_id': is_id, 'replicon': 'chr', 'side': side, 'locus_tag': tag, 'cds_start': start, 'cds_end': end,
            'cds_strand': strand, 'partial': 'False', 'pseudo': 'False', 'distance_bp': distance, 'effect': effect}


def flank(gene, side, group='ISA', dist=10, hops=0, effect=Effect.NONE, contig='ctg'):
    return Flank(group, side, gene, contig, hops, dist, effect, False)


def truth(side, tag, start, end, group='ISA', distance=10, effect='none'):
    return TruthRelation(group, side, tag, 'chr', start, end, '+', False, False, distance, effect)


def placed(**genes):
    table = {g: [Placement('chr', s, e, '+', 1.0)] for g, (s, e) in genes.items()}
    return lambda gene_id: table.get(gene_id, [])


def test_report_rows_become_flanks_nearest_first_and_passengers_crossing_an_end_count(tmp_path):
    path = tmp_path / 'r_report.tsv'
    write_tsv(path, REPORT_COLUMNS, [
        report_row('L1', 'IS1A_IS1_unknown_stitched,IS1F_IS1_unknown', 'my_ctg_9_100_400', 'OVERLAPPING_START',
                   effect='TRUNCATED|DISRUPTED'),
        report_row('L1', 'IS1A_IS1_unknown_stitched', 'my_ctg_9_900_1200', 'INSIDE'),
        report_row('L1', 'IS1A_IS1_unknown_stitched', '7_10_300', 'DOWNSTREAM', dist=40, hops=1),
        report_row('L2', 'IS26_IS6_unknown', 'None', 'NONE', dist=-1),
    ])
    loci = read_report_flanks(path, lambda name: name.split('_')[0])

    assert loci[1] == []  # an IS with no flanks is still a locus
    up, down = loci[0]
    assert (up.group, up.side, up.contig, up.overlapping, up.effect) == \
        ('IS1A', 'upstream', 'my_ctg_9', True, Effect.TRUNCATED | Effect.DISRUPTED)
    assert (down.side, down.contig, down.hops, down.distance_bp) == ('downstream', '7', 1, 40)


def test_cds_sequences_are_filed_under_the_report_gene_id(tmp_path):
    # eris names each record by the report's 0-based gene id; a frameshift-corrected call is longer than its coordinates
    path = tmp_path / 'r_cds.fna'
    path.write_text('>my_ctg_9_10_19 strand=+\nATGAAATAA\n>my_ctg_9_30_39 strand=-\nATGAAACTAA\n')
    assert read_cds_sequences(path) == {'my_ctg_9_10_19': b'ATGAAATAA', 'my_ctg_9_30_39': b'ATGAAACTAA'}


def test_cds_sequences_of_an_older_pyfgs_file_are_read_from_its_1_based_location(tmp_path):
    path = tmp_path / 'r_cds.fna'
    path.write_text('>ctg_FGS_1 ctg:11-19 strand=+\nATGAAATAA\n>ctg_FGS_2 ctg:31-39 strand=-\nATGAAACTAA\n')
    assert read_cds_sequences(path) == {'ctg_10_19': b'ATGAAATAA', 'ctg_30_39': b'ATGAAACTAA'}


def test_a_flank_on_the_wrong_side_counts_against_both_sides():
    truths = [truth('upstream', 'T1', 1000, 2000), truth('downstream', 'T2', 5000, 6000)]
    per_locus = [[flank('g1', 'downstream'), flank('g2', 'upstream'), flank('g3', 'downstream', contig='c2', hops=1)]]
    # g1 is T1's gene read on the wrong side; g2 matches nothing; g3 is T2
    score = score_flanks(truths, [[per_locus[0][0], per_locus[0][1]], [per_locus[0][2]]],
                         placed(g1=(1000, 2000), g2=(8000, 9000), g3=(5000, 6000)))

    _, matrix = side_confusion(score)
    assert matrix['upstream'] == {'downstream': 1}
    assert matrix['downstream'] == {'downstream': 1}
    assert matrix['no_truth'] == {'upstream': 1}
    assert sorted(r['status'] for r in relation_rows(score)) == ['FN_wrong_side', 'FP', 'FP_wrong_side', 'TP']

    values = {(m['side'], m['metric']): m['value'] for m in metrics(score)
              if m['dimension'] == 'relation' and m['replicon'] == 'all'}
    assert (values['upstream', 'tp'], values['upstream', 'fn_wrong_side'], values['upstream', 'fp']) == (0, 1, 1)
    assert (values['downstream', 'tp'], values['downstream', 'fp']) == (1, 1)
    assert values['both', 'precision'] == score.precision == 1 / 3
    assert values['both', 'recall'] == 1 / 2
    assert values['both', 'pooled_recall'] == score.pooled_recall == 1.0


def test_tandem_copies_flanked_by_one_cds_are_one_relation():
    truths = [truth('downstream', 'T1', 5000, 6000, distance=451), truth('downstream', 'T1', 5000, 6000, distance=256)]
    score = score_flanks(truths, [[flank('g1', 'downstream', dist=250)]], placed(g1=(5000, 6000)))

    assert score.total == {'upstream': 0, 'downstream': 1}
    assert [r.truth.distance_bp for r in score.truth_rows] == [256]


def test_disrupted_metrics_leave_out_unscored_effects():
    truths = [truth('upstream', 'T1', 1000, 2000, effect='disrupted'), truth('downstream', 'T2', 5000, 6000, effect=None),
              truth('upstream', 'T3', 9000, 9500, group='ISB')]
    per_locus = [[flank('g1', 'upstream', effect=Effect.TRUNCATED), flank('g2', 'downstream', effect=Effect.DISRUPTED)],
                 [flank('g3', 'upstream', group='ISB', effect=Effect.DISRUPTED)]]
    score = score_flanks(truths, per_locus, placed(g1=(1000, 2000), g2=(5000, 6000), g3=(9000, 9500)))

    values = {m['metric']: m['value'] for m in metrics(score) if m['dimension'] == 'disrupted' and m['replicon'] == 'all'}
    assert (values['called'], values['called_true'], values['tp'], values['truth']) == (2, 1, 1, 1)
    assert (values['precision'], values['recall']) == (score.disrupted_precision, score.disrupted_recall) == (0.5, 1.0)


def test_eris_validate_writes_the_tables_from_eris_output_files(tmp_path):
    rng = random.Random(29)
    chrom = ''.join(rng.choice('ACGT') for _ in range(20000))
    (tmp_path / 'ref.fasta').write_text(f'>chr\n{chrom}\n')
    (tmp_path / 'IS.csv').write_text('N°,Name,Family,Group,Synomyns,Iso,Origin\n45,IS1B,IS1,,IS1C,IS1A,E. coli\n')
    # Truth: IS1A_1 sits at 3000-4000 on +, between an upstream CDS on + and a downstream CDS on -
    write_tsv(tmp_path / 'truth.tsv', TRUTH_COLUMNS, [
        truth_row('IS1A_1', 'upstream', 'UP', 2000, 2900, distance=100),
        truth_row('IS1A_1', 'downstream', 'DOWN', 4020, 5000, strand='-', distance=20)])
    # eris called the IS IS1B (a synonym) on contig 'c_1', which holds chr[1000:6000]
    (tmp_path / 'r_cds.fna').write_text(f'>c_1_1000_1900 strand=+\n{chrom[2000:2900]}\n'
                                        f'>c_1_3020_4000 strand=-\n{reverse_complement(chrom[4020:5000])}\n')
    write_tsv(tmp_path / 'r_report.tsv', REPORT_COLUMNS, [
        report_row('L1', 'IS1B_IS1_unknown', 'c_1_1000_1900', 'UPSTREAM', dist=100),
        report_row('L1', 'IS1B_IS1_unknown', 'c_1_3020_4000', 'DOWNSTREAM', dist=20)])
    out = tmp_path / 'out'

    main(['--report', str(tmp_path / 'r_report.tsv'), '--cds', str(tmp_path / 'r_cds.fna'),
          '--reference', str(tmp_path / 'ref.fasta'), '--truth', str(tmp_path / 'truth.tsv'),
          '--synonyms', str(tmp_path / 'IS.csv'), '-o', str(out)])

    with open(out / 'relations.tsv') as handle:
        assert [r['status'] for r in csv.DictReader(handle, delimiter='\t')] == ['TP', 'TP']
    with open(out / 'metrics.tsv') as handle:
        f1 = [r['value'] for r in csv.DictReader(handle, delimiter='\t')
              if (r['dimension'], r['side'], r['replicon'], r['metric']) == ('relation', 'both', 'all', 'f1')]
    assert f1 == ['1.0000']
    assert {p.name for p in out.iterdir()} == {'metrics.tsv', 'relations.tsv', 'side_confusion.tsv',
                                               'distance_confusion.tsv', 'effect_confusion.tsv'}


def test_eris_validate_stops_when_the_report_and_cds_file_are_from_different_runs(tmp_path):
    (tmp_path / 'ref.fasta').write_text('>chr\nACGT\n')
    (tmp_path / 'IS.csv').write_text('N°,Name,Family,Group,Synomyns,Iso,Origin\n')
    write_tsv(tmp_path / 'truth.tsv', TRUTH_COLUMNS, [])
    (tmp_path / 'r_cds.fna').write_text('>c_0_9 strand=+\nATGAAATAA\n')
    write_tsv(tmp_path / 'r_report.tsv', REPORT_COLUMNS, [report_row('L1', 'IS1B_IS1_unknown', 'c_50_80', 'UPSTREAM')])

    with pytest.raises(SystemExit):
        main(['--report', str(tmp_path / 'r_report.tsv'), '--cds', str(tmp_path / 'r_cds.fna'),
              '--reference', str(tmp_path / 'ref.fasta'), '--truth', str(tmp_path / 'truth.tsv'),
              '--synonyms', str(tmp_path / 'IS.csv'), '-o', str(tmp_path / 'out')])
