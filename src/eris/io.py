"""
Module to handle query (contigs) and target (features) IO.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Union, IO, Iterator, Optional, Iterable
from pathlib import Path
from re import compile as re_compile
from gzip import open as gzopen
from bz2 import open as bzopen
from lzma import open as lzopen
from sys import stdout, stderr

from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.SeqRecord import SeqRecord

from mappy import Aligner

from pyfgs import FaaWriter, FnaWriter, Gff3Writer

from eris.alignment import Cigar
from eris.graph import Edge
from eris.interval import Strand
from eris.constants import FeatureType


# IS synonym helpers ---------------------------------------------------------------------------------------------------

_IS_BASE_RE = re_compile(r'^(IS\d+)[A-Za-z]*$')


def _is_element_base(target_name: str) -> str:
    """Return the canonical IS element base for a target name.

    Extracts the element identifier (first '_'-delimited field) and strips
    any trailing letter suffix so that variant names collapse to their family root.

    Examples::

        'IS30D_IS30_unknown'  → 'IS30'
        'IS1A_IS1_unknown'    → 'IS1'
        'IS30_IS30_unknown'   → 'IS30'
        'IS1203_IS3_IS51'     → 'IS1203'  (no suffix → unchanged)
    """
    element = target_name.split('_')[0]
    m = _IS_BASE_RE.match(element)
    return m.group(1) if m else element


def _group_synonyms(names: Iterable[str]) -> dict[str, list[str]]:
    """Group IS target names by their canonical base IS element.

    Returns a mapping of base_name → list-of-target-names for every group
    that contains more than one member (i.e. genuine synonym sets).
    Single-member groups are omitted.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        base = _is_element_base(name)
        groups.setdefault(base, []).append(name)
    return {base: members for base, members in groups.items() if len(members) > 1}


# Gene output filtering (_cds.fna, _proteins.faa) ----------------------------------------------------------------------

_FNA_HEADER_RE = re_compile(
    r'^>(?P<idtok>\S+)\s+(?P<contig>.+):(?P<start>\d+)-(?P<end>\d+)\s+strand=(?P<strand>[+-])\s*$'
)
_FNA_IDTOK_RE = re_compile(r'^(?P<base>.*)_FGS_(?P<n>\d+)$')
_HEADER_UNSAFE_RE = re_compile(r'[^\w.-]')


def _sanitize_header_token(token: str) -> str:
    """Replaces characters unsafe/confusing in a FASTA header id (e.g. '/') with '_'."""
    return _HEADER_UNSAFE_RE.sub('_', token)


def _load_locus_lookup(report_path: str) -> Optional[dict[str, tuple[str, str]]]:
    """Builds a `{contig}_{start}_{end}` (0-based) -> (target, context) lookup from `_report.tsv`.

    Returns None if the report is missing or malformed; an empty dict if it has no usable rows.
    """
    if not Path(report_path).exists():
        return None

    lookup: dict[str, tuple[str, str]] = {}
    with open(report_path) as fh:
        header_cols = fh.readline().rstrip("\n").split("\t")
        try:
            gid_i = header_cols.index("gene_id")
            ctx_i = header_cols.index("context")
            tgt_i = header_cols.index("target")
        except ValueError:
            return None
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(gid_i, ctx_i, tgt_i):
                continue
            context = fields[ctx_i]
            if context == "NONE":
                continue
            lookup.setdefault(fields[gid_i], (_sanitize_header_token(fields[tgt_i]), context))

    return lookup


def _rewrite_fgs_headers(path: str, lookup: dict[str, tuple[str, str]]) -> None:
    """Filters and rewrites pyfgs-style FASTA records (`_cds.fna` or `_proteins.faa`) in place.

    Keeps only records whose coordinates match a locus-related gene in `lookup` (i.e. genes
    reported in `_report.tsv` as INSIDE/flanking an IS), dropping all other assembly-wide CDS.
    Kept headers are named `{target}_{context}_{contig}_FGS_{n}` (n = pyfgs's per-contig gene
    index, so the contig keeps names unique) and display 0-based coordinates
    (matching `_report.tsv`'s gene_id) instead of pyfgs's default 1-based display.
    """
    keep = False
    tmp_path = f"{path}.tmp"
    with open(path) as src, open(tmp_path, "w") as dst:
        for line in src:
            if not line.startswith(">"):
                if keep:
                    dst.write(line)
                continue

            m = _FNA_HEADER_RE.match(line.rstrip("\n"))
            if not m:
                keep = False
                continue

            idtok, contig, start, end, strand = m.group("idtok", "contig", "start", "end", "strand")
            start0 = int(start) - 1
            hit = lookup.get(f"{contig}_{start0}_{end}")
            keep = hit is not None
            if not keep:
                continue

            idm = _FNA_IDTOK_RE.match(idtok)
            n = idm.group("n") if idm else "1"
            dst.write(f">{hit[0]}_{hit[1]}_{_sanitize_header_token(contig)}_FGS_{n} {contig}:{start0}-{end} strand={strand}\n")

    Path(tmp_path).replace(path)


def _filter_gene_outputs(prefix: str) -> None:
    """Reduces `{prefix}_cds.fna` and `{prefix}_proteins.faa` to the IS-related genes listed in
    `{prefix}_report.tsv`, tagging their headers with locus target/context (0-based coordinates).
    `{prefix}_assembly.gff` is left untouched and keeps all assembly-wide genes.

    Best-effort: any failure is logged to stderr and leaves the files untouched rather
    than crashing the pipeline after it has already produced valid output.
    """
    try:
        _do_filter_gene_outputs(prefix)
    except Exception as e:
        stderr.write(f"[eris] _filter_gene_outputs failed ({type(e).__name__}: {e}); gene files left unchanged\n")


def _do_filter_gene_outputs(prefix: str) -> None:
    report_path = f"{prefix}_report.tsv"
    lookup = _load_locus_lookup(report_path)
    if lookup is None:
        return

    for path in (f"{prefix}_cds.fna", f"{prefix}_proteins.faa"):
        if not Path(path).exists():
            continue
        _rewrite_fgs_headers(path, lookup)


# Base classes ---------------------------------------------------------------------------------------------------------
class _Handle:
    def __init__(self, handle: IO):
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handle.close()

# Readers --------------------------------------------------------------------------------------------------------------
class GfaReader(_Handle):
    """
    Reader for Graphical Fragment Assembly (GFA) files.

    Parses Segment (S) and Link (L) lines into SeqRecord and Edge objects.
    """

    @staticmethod
    def _parse_segment(parts: list[str]):
        name = parts[0]
        seq = parts[1]
        tags = {}
        for item in parts[2:]:
            tag, typ, val = item.split(':', maxsplit=2)
            if typ == 'f':
                val = float(val)
            elif typ == 'i':
                val = int(val)
            tags[tag] = val
        return SeqRecord(seq=Seq(seq), id=name, name=name, annotations=tags)

    @staticmethod
    def _parse_link(parts: list[str]):
        u = parts[0]
        u_strand = Strand(parts[1])
        v = parts[2]
        v_strand = Strand(parts[3])
        cigar = Cigar(parts[4])
        overlap = next((n for op, n, _, _, _ in cigar if op == 'M'), 0)
        return Edge(u, u_strand, v, v_strand, overlap)

    @classmethod
    def _parse_line(cls, line):
        if line.startswith('S\t'):
            return cls._parse_segment(line[2:].rstrip().split('\t'))
        elif line.startswith('L\t'):
            return cls._parse_link(line[2:].rstrip().split('\t'))
        else:
            return None

    def __next__(self):
        while True:  # Will naturally raise StopIteration when the handle is exhausted
            if (parsed := self._parse_line(next(self._handle))) is not None:
                return parsed

    def __iter__(self):
        for line in self._handle:
            if (parsed := self._parse_line(line)) is not None:
                yield parsed


# Writers --------------------------------------------------------------------------------------------------------------
class GfaWriter(_Handle):
    """Writer for GFA format files."""

    def write(self, item: Union[Edge, SeqRecord]) -> int:
        """Writes an Edge (Link) or SeqRecord (Segment) to the file."""
        if isinstance(item, Edge):
            return self._handle.write(f"L\t{item.u}\t{item.u_strand}\t{item.v}\t{item.v_strand}\t*\n")
        elif isinstance(item, SeqRecord):
            return self._handle.write(f"S\t{item.id}\t{item.seq}\n")
        raise TypeError(f"Unsupported type: {type(item)}")


# Classes --------------------------------------------------------------------------------------------------------------
class TargetDatabase:
    """
    Manages a database of nucleotide target sequences (e.g. MGEs, ARGs).

    Wraps a minimap2 (mappy) index for efficient searching.

    Example:
        >>> db = TargetDatabase("targets.fasta")
        >>> aligner = db.aligner
    """
    __slots__ = ('path', 'feature_type', 'indexing_threads', '_aligner')

    def __init__(self, path: Union[str, Path], feature_type: FeatureType = FeatureType.CDS, indexing_threads: int = 3):
        """
        Initialize the TargetDatabase.

        Args:
            path: Path to the FASTA or MMI file.
            feature_type: Classification for these targets.
            indexing_threads: Threads to use for on-the-fly indexing.
        """
        self.path = Path(path)
        self.feature_type = feature_type
        self.indexing_threads = indexing_threads  # Only needed if not already indexed
        self._aligner = None

        if not self.path.exists():
            raise FileNotFoundError(f"Target database not found: {self.path}")

    @property
    def aligner(self) -> Aligner:
        """Lazy-loaded mappy.Aligner instance."""
        if self._aligner is None:
            self._aligner = Aligner(fn_idx_in=str(self.path), n_threads=self.indexing_threads)
            if not self._aligner:
                raise ValueError(f"Minimap2 failed to load database: {self.path}")
        return self._aligner


@dataclass(slots=True, frozen=True)
class GenomeAssembly:
    """
    Container for a genome assembly, including contigs and their graph topology.

    Handles FASTA and GFA formats, with support for transparent decompression.

    Example:
        >>> assembly = GenomeAssembly.from_file("assembly.gfa.gz")
        >>> for contig_id, seq in assembly:
        >>>     print(contig_id, len(seq))
    """
    _SEQUENCE_FILE_REGEX = re_compile(
        r'\.('
        r'(?P<fasta>f(asta|a|na|fn|as|aa))|'
        r'(?P<gfa>gfa)|'
        r')\.?(?P<compression>(gz|bz2|xz))?$'
    )
    _OPENERS = {'gz': gzopen, 'bz2': bzopen, 'xz': lzopen}
    id: str
    contigs: dict[str, Seq]
    edges: list[Edge]
    contig_depths: dict[str, float]
    contig_lengths: dict[str, int]

    def __len__(self):
        """Total number of base pairs in the assembly."""
        return sum(len(i) for i in self.contigs.values())

    def __iter__(self) -> Iterator[tuple[str, Seq]]:
        """Iterate over contig IDs and sequences."""
        return iter(self.contigs.items())

    def __str__(self):
        return self.id

    def __getitem__(self, item: str) -> 'Seq':
        """Access a contig sequence by its ID."""
        return self.contigs[item]

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """
        Load an assembly from a FASTA or GFA file.

        Args:
            file: Path to the file. Supports .gz, .bz2, and .xz compression.
        """
        file = Path(file) # type: Path
        if not (m := cls._SEQUENCE_FILE_REGEX.search(file.name)):
            raise NotImplementedError(f'Unsupported format: {file}')
        reader = FastaIterator if m.group('fasta') else GfaReader
        with cls._OPENERS.get(m.group('compression'), open)(file, mode='rt') as handle:
            return cls.from_stream(handle, reader, file.name.rstrip(m.group()))

    @classmethod
    def from_stream(cls, handle: IO[str], reader, id_: str = None):
        """Load an assembly from an open file stream using the specified reader."""
        contigs, edges, depths, lengths = {}, [], {}, {}
        for record in reader(handle):
            if isinstance(record, SeqRecord):
                contigs[record.id] = record.seq
                depths[record.id] = record.annotations.get('DP', record.annotations.get('depth', 1.0))
                lengths[record.id] =  len(record)
            elif isinstance(record, Edge):
                edges.append(record)
        return cls(id_ or handle.name, contigs, edges, depths, lengths)


@dataclass(slots=True, frozen=True)
class ReportRow:
    """
    Represents a single structural variant or passenger gene record in the eris TSV report.

    Attributes:
        locus_id: Unique identifier for the assembled structural variant locus.
        target: The mobile genetic element(s) or query sequences found in this locus.
        gene_id: The identifier of the contextual passenger or flanking gene.
        context: The spatial relationship (e.g., INSIDE, UPSTREAM) of the gene to the target.
        dist_bp: Distance in base pairs between the gene and the target element.
        topo_hops: Number of graph nodes traversed to find this gene (0 if on the same contig).
        orientation: Strand orientation of the gene relative to the target (same or opposite).
        effect: Biological impact of the insertion on the gene (e.g., TRUNCATED, NONE).
        fractional_depth: The relative read depth of the variant path compared to the source contig, indicating sub-clonal abundance.
        ref_contig: Reference sequence name after coordinate liftover (requires --reference).
        ref_start: Locus start in reference coordinates (requires --reference).
        ref_end: Locus end in reference coordinates (requires --reference).
        coverage_pct: % of the target (IS) reference covered by its alignments in this locus; a partial
            match (e.g. a truncated IS) has a low value.
    """
    locus_id: str
    target: str
    gene_id: str
    context: str
    dist_bp: int
    topo_hops: int
    orientation: str
    effect: str
    fractional_depth: float
    locus_contig: str = "."
    locus_start: int = -1
    locus_end: int = -1
    coverage_pct: float = -1.0

    @classmethod
    def header(cls, mode: str = 'variant') -> str:
        """Returns the TSV header string dynamically generated from the dataclass fields."""
        fields = list(cls.__annotations__.keys())

        # Swap the header text if we are calculating copy numbers
        if mode == 'collapse' and 'fractional_depth' in fields:
            fields[fields.index('fractional_depth')] = 'copy_number'

        return "\t".join(fields) + "\n"

    def to_tsv(self) -> str:
        """Formats the row data into a tab-separated string."""
        return (f"{self.locus_id}\t{self.target}\t{self.gene_id}\t{self.context}\t"
                f"{self.dist_bp}\t{self.topo_hops}\t{self.orientation}\t{self.effect}\t"
                f"{self.fractional_depth}\t{self.locus_contig}\t{self.locus_start}\t{self.locus_end}\t"
                f"{self.coverage_pct}\n")




class OutputManager:
    """
    Manages all file outputs and stream lifecycles for the eris pipeline.

    Handles writing of the TSV report, GFF3 annotations, protein FASTA,
    and locus nucleotide sequences.

    Example:
        >>> with OutputManager("my_sample") as out:
        >>>     out.write_locus_relations(locus)
    """

    def __init__(self, prefix: Optional[str], write_gff: bool = True, write_faa: bool = True,
                 write_fna: bool = True, mode: str = 'variant',
                 fasta_lookup: Optional[dict] = None):
        """
        Initialize the OutputManager.

        Args:
            prefix: Filename prefix for all output files. If None, TSV is sent to stdout.
            write_gff: Whether to output an assembly-wide GFF3 file.
            write_faa: Whether to output a protein FASTA of IS-related genes.
            mode: Pipeline mode ('variant' or 'collapse') to determine TSV headers.
            fasta_lookup: target_name → [(ref_contig, ref_start, ref_end, ref_seq), ...] built by running LinearPipeline on a reference FASTA (-ar flag). When provided,
            a separate *_ref_positions.tsv is written with one row per IS copy found in the reference.
        """
        self.prefix = prefix
        self.write_gff = write_gff
        self.write_fna = write_fna
        self.write_faa = write_faa
        self.mode = mode
        self.fasta_lookup = fasta_lookup or {}
        self._stack = ExitStack()

    def __enter__(self):
        # 1. Setup TSV Report (Stdout or File)
        if self.prefix:
            self.tsv_handle = self._stack.enter_context(open(f"{self.prefix}_report.tsv", "w"))
        else:
            self.tsv_handle = stdout

        # Write TSV Header dynamically from the dataclass
        self.tsv_handle.write(ReportRow.header(self.mode))

        # 2. Setup Optional PyFGS Writers
        self.gff_writer = None
        self.faa_writer = None
        self.fna_writer = None

        if self.prefix and self.write_gff:
            self.gff_writer = self._stack.enter_context(Gff3Writer(f"{self.prefix}_assembly.gff"))

        if self.prefix and self.write_faa:
            self.faa_writer = self._stack.enter_context(FaaWriter(f"{self.prefix}_proteins.faa"))

        if self.prefix and self.write_fna:
            self.fna_writer = self._stack.enter_context(FnaWriter(f"{self.prefix}_cds.fna"))

        # 3. Setup Locus FASTA writer
        self.locus_fasta = None
        if self.prefix:
            self.locus_fasta = self._stack.enter_context(open(f"{self.prefix}_loci.fasta", "w"))

        # 4. Setup reference-positions writer (only when fasta_lookup was provided)
        self._ref_pos_handle = None
        if self.prefix and self.fasta_lookup:
            self._ref_pos_handle = self._stack.enter_context(
                open(f"{self.prefix}_ref_positions.tsv", "w")
            )
            self._ref_pos_handle.write(
                "target\tref_contig\tref_start\tref_end\tgfa_count\tref_count\tstatus\n"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self.prefix and (self.write_fna or self.write_faa):
            _filter_gene_outputs(self.prefix)

    # --- Helper methods ---

    def write_locus_relations(self, locus: 'Locus'):
        """Writes all contextual relationships (passengers, flanks) for a locus to the TSV report."""

        relations = locus.passengers + locus.upstream_flanks + locus.downstream_flanks
        # If there are no flanking genes, log the isolated IS anyway
        if not relations:
            row = ReportRow(
                locus_id=locus.id,
                target=",".join(t.id for t in locus.targets),
                gene_id="None",
                context="NONE",
                dist_bp=-1,
                topo_hops=0,
                orientation="-",
                effect="NONE",
                fractional_depth=locus.fractional_depth,
                locus_contig=locus.contig,
                locus_start=locus.start,
                locus_end=locus.end,
                coverage_pct=locus.coverage_pct,
            )
            self.tsv_handle.write(row.to_tsv())
        else:
            for relation in relations:
                self._write_tsv_row(locus, relation)

    def _write_tsv_row(self, locus: 'Locus', relation: 'FeatureRelation'):
        """Constructs a ReportRow dataclass and writes it to the TSV handle."""

        row = ReportRow(
            locus_id=locus.id,
            target=",".join(t.id for t in locus.targets),
            gene_id=relation.feature.id,
            context=relation.spatial.name,
            dist_bp=relation.distance_bp,
            topo_hops=relation.topological_dist,
            orientation=relation.orientation.value,
            effect=relation.effect.name,
            fractional_depth=locus.fractional_depth,
            locus_contig=locus.contig,
            locus_start=locus.start,
            locus_end=locus.end,
            coverage_pct=locus.coverage_pct,
        )
        self.tsv_handle.write(row.to_tsv())

    def write_ref_positions(self, gfa_counts: dict[str, int]):
        """Cross-reference GFA IS counts against every IS in the reference.

        Covers three cases per IS name:
          - found in both   → one row per reference copy, with gfa_count / ref_count
          - ref only        → one row per reference copy, gfa_count=0  (not called in GFA)
          - GFA only        → one row with ref_contig/start/end='.', ref_count=0

        The 'status' column summarises the discrepancy at a glance:
          consistent   – same count on both sides
          discrepant   – counts differ (investigate which copies are missing)
          not_in_gfa   – present in reference but GFA never called it
          gfa_only     – called by GFA but absent from reference
        """
        if not self._ref_pos_handle:
            return

        all_names = sorted(set(gfa_counts) | set(self.fasta_lookup))

        # Identify synonym groups where pooling resolves a one-sided miss.
        # Only revise status when at least one member would be not_in_gfa / gfa_only
        # and the pooled counts are non-zero on both sides.
        synonym_status: dict[str, str] = {}
        for _base, group_names in _group_synonyms(all_names).items():
            indiv = []
            for n in group_names:
                g = gfa_counts.get(n, 0)
                r = len(self.fasta_lookup.get(n, []))
                if g == 0:
                    indiv.append("not_in_gfa")
                elif r == 0:
                    indiv.append("gfa_only")
                elif g == r:
                    indiv.append("consistent")
                else:
                    indiv.append("discrepant")

            if "not_in_gfa" in indiv or "gfa_only" in indiv:
                pooled_gfa = sum(gfa_counts.get(n, 0) for n in group_names)
                pooled_ref = sum(len(self.fasta_lookup.get(n, [])) for n in group_names)
                if pooled_gfa > 0 and pooled_ref > 0:
                    revised = "consistent_synonym" if pooled_gfa == pooled_ref else "discrepant_synonym"
                    for n in group_names:
                        synonym_status[n] = revised

        for name in all_names:
            ref_hits = self.fasta_lookup.get(name, [])
            gfa_count = gfa_counts.get(name, 0)
            ref_count = len(ref_hits)

            if name in synonym_status:
                status = synonym_status[name]
            elif gfa_count == 0:
                status = "not_in_gfa"
            elif ref_count == 0:
                status = "gfa_only"
            elif gfa_count == ref_count:
                status = "consistent"
            else:
                status = "discrepant"

            if ref_hits:
                for ref_ctg, ref_st, ref_en, _seq in ref_hits:
                    self._ref_pos_handle.write(
                        f"{name}\t{ref_ctg}\t{ref_st}\t{ref_en}\t{gfa_count}\t{ref_count}\t{status}\n"
                    )
            else:
                # GFA-only: no reference position to report
                self._ref_pos_handle.write(
                    f"{name}\t.\t.\t.\t{gfa_count}\t{ref_count}\t{status}\n"
                )

    def write_global_genes(self, contig_id: str, sequence: Union[str, bytes], genes: list['Gene']):
        """Writes assembly-wide gene calls to the GFF3 and FAA files."""
        if self.gff_writer:
            self.gff_writer.write_record(genes, contig_id, sequence)
        if self.faa_writer:
            self.faa_writer.write_record(genes, contig_id)
        if self.fna_writer:
            self.fna_writer.write_record(genes, contig_id)

    def write_locus_fasta(self, locus_id: str, sequence: str):
        """Writes a single locus nucleotide sequence to the loci FASTA file."""
        if self.locus_fasta:
            self.locus_fasta.write(f">{locus_id}\n{sequence}\n")