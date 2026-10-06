"""
Module for managing the CLI layer on top of the API; also contains the CLI entry point under `main()`.
"""
import argparse
import dataclasses
from pathlib import Path
from sys import stderr

from eris._version import __version__
from eris.constants import FeatureType


# Classes --------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(slots=True, frozen=True)
class Log:
    quiet: bool = False
    stream = stderr

    def msg(self, msg: str, flush: bool = False) -> None:
        if self.quiet: return
        self.stream.write(msg)
        if flush: self.stream.flush()


# Main CLI entry-point -------------------------------------------------------------------------------------------------
def main():

    # Define args ------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Graph-aware contextual annotation of targeted genomic features',
        usage="%(prog)s -i genome.gfa -d targets.fasta [options]", add_help=False, prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group('📁', 'Input arguments')
    inputs.add_argument('-i', '--genome', type=Path, required=True, metavar='',
                        help='Path to a genome in fasta or GFA format; File may be compressed.')
    inputs.add_argument('-d', '--targets', type=Path, required=True, metavar='',
                        help='Path to target nucleotide features in fasta or mmi format')
    inputs.add_argument('-ar', '--assembly-reference', type=Path, default=None, metavar='',
                        help='Complete reference genome FASTA (e.g. a finished chromosome). '
                             'LinearPipeline is run on it to find IS locations; those coordinates '
                             'populate ref_contig/ref_start/ref_end in the report (GFA input only).')
    outs = parser.add_argument_group('💾', 'Output arguments')
    outs.add_argument('-o', '--outprefix', type=str, default=None, metavar='',
                      help='Prefix for output files (if absent, prints TSV to stdout)')
    outs.add_argument('--no-gff', action='store_true',
                      help='Do not write GFF3 output for global genes')
    outs.add_argument('--no-faa', action='store_true',
                      help='Do not write FAA output for global proteins')
    outs.add_argument('--no-fna', action='store_true',
                      help='Do not write FNA output for global transcripts (CDS)')

    pipeline_args = parser.add_argument_group('⚙️', 'Pipeline arguments')
    pipeline_args.add_argument('--hops', type=int, default=3, metavar='',
                        help='Maximum number of contextual genes to sweep upstream/downstream')
    pipeline_args.add_argument('-l', '--tolerance', type=int, default=0, metavar='',
                        help='Distance tolerance (bp) for merging clustered target alignments')
    pipeline_args.add_argument('-t', '--max-workers', type=int, default=None, metavar='',
                        help='Maximum number of worker threads for alignment and CDS prediction')
    pipeline_args.add_argument('-m', '--mode', choices=['variant', 'collapse'], default='variant',
                        help='Resolution mode: "variant" for all possible structural path, "collapse" for a consensus')
    pipeline_args.add_argument('-c', '--min-coverage', type=float, default=0.30, metavar='',
                        help='Minimum coverage fraction (0.0 to 1.0 required to report a locus target)')
    pipeline_args.add_argument('--min-identity', type=float, default=0.75, metavar='',
                        help='Minimum sequence identity (0.0 to 1.0) to keep an IS alignment; '
                             'lower values recover diverged IS but may increase false positives')

    targets = parser.add_argument_group('🎯', 'Target arguments')
    targets.add_argument('-f', '--feature-type', choices=[e.value for e in FeatureType], metavar='', 
                         default=FeatureType.CDS.value, help='Type of feature to annotate')
    targets.add_argument('--indexing-threads', type=int, default=3, metavar='',
                        help='Number of threads to use for indexing if not already indexed')

    opts = parser.add_argument_group('🛠️', 'Other options')
    opts.add_argument('-q', '--quiet', action='store_true',
                      help='Suppress console logging output')
    opts.add_argument('-v', '--version', action='version', version=__version__,
                      help='Show version number and exit')
    opts.add_argument('-h', '--help', action='help',
                      help='Show this help message and exit')

    # Parse args -------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    log = Log(quiet=args.quiet)

    # Load inputs ------------------------------------------------------------------------------------------------------
    from eris.io import GenomeAssembly, TargetDatabase, OutputManager

    log.msg("🎯 Loading targets...\n", flush=True)
    target_db = TargetDatabase(args.targets, FeatureType(args.feature_type), args.indexing_threads)
    log.msg("✅ Targets loaded!\n")

    log.msg("🧬 Loading genome...\n", flush=True)
    genome = GenomeAssembly.from_file(args.genome)
    log.msg("✅ Genome loaded!\n")

    # Run pipeline -----------------------------------------------------------------------------------------------------
    from eris.pipeline import Pipeline, LinearPipeline

    # Put all output files in one folder, keeping the same basename inside it
    outdir = Path(args.outprefix)
    outdir.mkdir(parents=True, exist_ok=True)
    outprefix = outdir / outdir.name

    # Decide which pipeline to use based on whether the input has graph topology
    has_topology = bool(genome.edges)

    if has_topology:
        pipeline_cls = Pipeline
        pipeline_kwargs = dict(
            max_feature_hops=args.hops,
            locus_tolerance=args.tolerance,
            max_workers=args.max_workers,
            mode=args.mode,
            min_coverage=args.min_coverage,
            min_identity=args.min_identity,
        )
        log_msg = "⌛️ Running topological traversal...\n"

    else:
        pipeline_cls = LinearPipeline
        pipeline_kwargs = dict(
            max_feature_hops=args.hops,
            locus_tolerance=args.tolerance,
            max_workers=args.max_workers,
            min_coverage=args.min_coverage,
            min_identity=args.min_identity,
        )
        log_msg = "⌛️ Running linear (non-graph) locus assembly...\n"

    # Build IS → reference coordinate index if -ar was given
    fasta_lookup = None
    if args.assembly_reference:
        if not has_topology:
            log.msg("⚠️  -ar requires GFA input; skipping reference IS mapping.\n")
        else:
            log.msg("🔬 Finding IS locations in reference assembly...\n", flush=True)
            ref_genome = GenomeAssembly.from_file(args.assembly_reference)
            fasta_lookup = {}
            with LinearPipeline(target_db, max_feature_hops=0,
                                 locus_tolerance=args.tolerance,
                                 max_workers=args.max_workers,
                                 min_coverage=args.min_coverage,
                                 min_identity=args.min_identity) as lin:
                for locus in lin(ref_genome):
                    ref_seq = locus.extract_sequence(ref_genome)
                    for target in locus.targets:
                        name = target.id.removesuffix('_stitched')
                        fasta_lookup.setdefault(name, []).append(
                            (locus.contig, locus.start, locus.end, ref_seq)
                        )
            n_types = len(fasta_lookup)
            n_sites = sum(len(v) for v in fasta_lookup.values())
            log.msg(f"✅ Reference indexed: {n_types} IS type(s) at {n_sites} site(s).\n")

    # Hook up the OutputManager parameters explicitly
    with (pipeline_cls(target_db, **pipeline_kwargs) as pipeline,
          OutputManager(outprefix, write_gff=not args.no_gff, write_faa=not args.no_faa,
                        write_fna=not args.no_fna, mode=args.mode,
                        fasta_lookup=fasta_lookup) as out):

        log.msg(log_msg, flush=True)

        gfa_counts: dict[str, int] = {}
        for locus in pipeline(genome, out=out):
            out.write_locus_relations(locus)
            for t in locus.targets:
                name = t.id.removesuffix('_stitched')
                gfa_counts[name] = gfa_counts.get(name, 0) + 1
            if out.locus_fasta:
                out.write_locus_fasta(locus.id, locus.extract_sequence(genome))

        out.write_ref_positions(gfa_counts)

    log.msg("🎉 Pipeline completed!\n")

if __name__ == "__main__":
    main()