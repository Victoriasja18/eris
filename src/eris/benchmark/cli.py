"""
`eris-validate`: scores an eris run's IS-CDS relations against a curated relation truth, from eris's output files.
"""
import argparse
from pathlib import Path
from sys import stderr

from eris.benchmark.relations import (
    GenePlacer,
    read_cds_sequences,
    read_report_flanks,
    read_truth,
    report,
    score_flanks,
)
from eris.benchmark.tables import write_tables


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog='eris-validate', description="Scores eris's upstream/downstream CDS, distances and effects against a "
                                          "curated relation truth, keyed by ISfinder synonym group")
    parser.add_argument('--report', type=Path, required=True, help="eris's {prefix}_report.tsv")
    parser.add_argument('--cds', type=Path, required=True, help="eris's {prefix}_cds.fna from the same run")
    parser.add_argument('--reference', type=Path, required=True, help='Reference genome FASTA the truth is on')
    parser.add_argument('--truth', type=Path, required=True, help='Relation truth TSV (e.g. ec958_IS_cds_relations.tsv)')
    parser.add_argument('-o', '--outdir', type=Path, required=True, help='Folder for the output tables')
    parser.add_argument('--reach-report', type=Path,
                        help='Report of the same assembly run with a larger --hops: a miss it finds is put down to the '
                             'hops budget rather than to a wrong gene call')
    parser.add_argument('--synonyms', type=Path, metavar='IS.csv',
                        help="ISfinder's IS.csv; downloaded and cached if not given")
    parser.add_argument('--replicons', help='Comma-separated replicons the assembly holds; truth relations on other '
                                            'replicons are left out (default: all)')
    args = parser.parse_args(argv)

    from eris.synonyms import SynonymResolver, fetch_isfinder_table
    group_of = SynonymResolver(args.synonyms or fetch_isfinder_table()).group

    replicons = set(args.replicons.split(',')) if args.replicons else None
    truths = read_truth(args.truth, group_of, replicons)
    per_locus = read_report_flanks(args.report, group_of)
    reach = read_report_flanks(args.reach_report, group_of) if args.reach_report else None

    # Gene calls do not depend on the hops budget, so the reach run's genes are in this run's _cds.fna too
    sequences = read_cds_sequences(args.cds)
    reported = {f.gene_id for flanks in (per_locus + (reach or [])) for f in flanks}
    if missing := sorted(reported - sequences.keys()):
        parser.error(f"{len(missing)} reported genes are not in {args.cds}, e.g. {missing[:3]}; "
                     f"are the report and the CDS file from the same run?")

    placer, placements = GenePlacer(args.reference), {}

    def place(gene_id):
        if gene_id not in placements:
            placements[gene_id] = placer.place(sequences[gene_id])
        return placements[gene_id]

    score = score_flanks(truths, per_locus, place, reach)
    write_tables(score, args.outdir)
    print(report(score), file=stderr)


if __name__ == '__main__':
    main()
