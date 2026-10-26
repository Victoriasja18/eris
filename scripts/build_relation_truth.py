"""
Builds the IS–CDS relation truth set for the EC958 benchmark.

Takes the hand-curated relations (which CDS lies downstream, upstream and inside each IS, and whether the insertion
disrupts it) and rebuilds every coordinate, strand, partial flag and distance from the RefSeq annotation, so typos
in the curated free text don't reach the benchmark. Relations whose recomputed values disagree with the curated
ones are written to a separate review file.

Only the curated downstream and upstream CDS are used. Genes inside an IS (its own transposase, the passengers) are
taken from the annotation instead: the curated Inside column has many copy-pasted locus tags from elsewhere in the
genome. Curated distances are only checked for gross errors (a wrong locus tag), since many were transcribed by hand
and disagree with the bedtools distances they came from.

Upstream and downstream are relative to the IS's strand, as in the curated table and in eris.

The curator's per-IS effect is kept in `curated_effect`. "Upregulates" is not scored (see
docs/adr/0002-relation-benchmark-truth.md), so its relations get effect `none`. A flank that lies inside another truth
IS (adjacent IS pairs such as IS30/ISSd1) is kept, with that IS in `inside_is`, so the benchmark can leave its
effect unscored: its overlap with the IS comes from the IS boundaries, not from an insertion.

Usage:
    uv run python scripts/build_relation_truth.py CURATED GFF BED [-o OUT] [-r REVIEW]

    CURATED  NZ_HG941718.1_IS_cds_effect.txt (chromosome NZ_HG941718.1 and plasmid pEC958 NZ_HG941719.1)
    GFF      GCF_000285655.3_EC958.v1_genomic.gff.gz
    BED      data/complete_ec958_IS.reanotated.sorted.bed (IS names and strands)
"""
import argparse
import csv
import gzip
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Locus tags in the curated cells, with or without their "(gene;strand;partial;Nbp)" note
TAG_PATTERN = re.compile(r'(E?C958_RS\d+)\s*(?:\(([^)]*)\)?)?')
CURATED_SIDES = ('downstream', 'upstream')
DISTANCE_TOLERANCE = 100  # bp between the curated and recomputed distance before a relation is reviewed
MAX_REACH = 10_000  # bp; a curated CDS further from its IS than this is almost certainly a wrong locus tag
TRUTH_FIELDS = ['is_id', 'replicon', 'is_start', 'is_end', 'is_strand', 'side', 'locus_tag', 'gene', 'product',
                'cds_start', 'cds_end', 'cds_strand', 'partial', 'pseudo', 'distance_bp', 'effect', 'curated_effect',
                'inside_is', 'note']
SCORED_EFFECTS = ('disrupts', 'no effect', 'upregulates')  # curated effects that are not review items


@dataclass
class Cds:
    replicon: str
    start: int  # 0-based, half-open, like the BED
    end: int
    strand: str
    partial: bool
    pseudo: bool
    gene: str = ''
    product: str = ''


def strip_refseq(replicon: str) -> str:
    """NZ_HG941718.1 -> HG941718.1, the name used by the benchmark FASTA and truth BED."""
    return replicon.removeprefix('NZ_')


def read_gff(path: Path) -> dict[str, Cds]:
    """Indexes genes and pseudogenes by locus tag; products come from their CDS lines."""
    opener = gzip.open if path.suffix == '.gz' else open
    genes, products = {}, {}
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            seqid, _, ftype, start, end, _, strand, _, attributes = line.rstrip('\n').split('\t')
            attrs = dict(a.split('=', 1) for a in attributes.split(';') if '=' in a)
            tag = attrs.get('locus_tag')
            if not tag:
                continue
            if ftype in ('gene', 'pseudogene'):
                genes[tag] = Cds(strip_refseq(seqid), int(start) - 1, int(end), strand,
                                 partial=attrs.get('partial') == 'true',
                                 pseudo=ftype == 'pseudogene' or attrs.get('pseudo') == 'true',
                                 gene=attrs.get('gene', ''))
            elif ftype == 'CDS':
                products.setdefault(tag, attrs.get('product', '').replace('%2C', ',').replace('%3B', ';'))
    for tag, cds in genes.items():
        cds.product = products.get(tag, '')
    return genes


def read_bed(path: Path) -> dict[tuple[str, int, int], tuple[str, str]]:
    """(replicon, start, end) -> (IS name, strand)."""
    with open(path) as handle:
        rows = (line.rstrip('\n').split('\t') for line in handle if line.strip())
        return {(r[0], int(r[1]), int(r[2])): (r[3], r[5]) for r in rows}


def read_curated(path: Path) -> list[dict]:
    """Reads the curated table, skipping its '##' description lines."""
    text = path.read_bytes().decode('utf-8', errors='replace')
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith(('"##', '##'))]
    header = [h for h in lines[0].lstrip('#').split('\t') if h]
    return [dict(zip(header + ['Note'], row)) for row in csv.reader(lines[1:], delimiter='\t', quotechar='"')]


def parse_cell(cell: str) -> list[tuple[str, bool | None, int | None]]:
    """Returns (locus tag, curated partial, curated distance) for each CDS in a curated cell; '*' means none."""
    entries = []
    for tag, note in TAG_PATTERN.findall(cell or ''):
        tag = 'E' + tag if tag.startswith('C958_') else tag
        note = note or ''
        partial = None if 'partial' not in note else 'no partial' not in note
        distance = re.search(r'(-?\d+)\s*(?:bp)?\s*$', note)
        entries.append((tag, partial, int(distance.group(1)) if distance else None))
    return entries


def relate(is_start: int, is_end: int, is_strand: str, cds: Cds) -> tuple[str, int]:
    """
    Places a CDS relative to an IS, reading in the IS's direction.
    Returns (side, distance): the gap in bp to the nearer IS end, negative when the CDS overlaps that end.
    """
    if cds.start >= is_start and cds.end <= is_end:
        return 'inside', 0
    if cds.start < is_start and cds.end > is_end:
        return 'spanning', -(is_end - is_start)
    # Which end of the IS (in genome coordinates) the CDS sits beyond, or overlaps
    if cds.start < is_start:
        genome_side, distance = 'left', is_start - cds.end
    else:
        genome_side, distance = 'right', cds.start - is_end
    downstream_side = 'right' if is_strand != '-' else 'left'
    return ('downstream' if genome_side == downstream_side else 'upstream'), distance


def relation_effect(row_effect: str, side: str, cds: Cds, distance: int) -> str:
    """
    The curated effect is per IS; it applies to a flanking CDS that the insertion split, which the annotation
    marks as partial or pseudo (or which overlaps the IS). The IS's own genes (inside) are never disrupted.
    Upregulation is not scored, so an upregulating IS's flanks get `none`.
    """
    row_effect = row_effect.strip().lower()
    if row_effect not in SCORED_EFFECTS:
        return 'uncertain'
    if row_effect == 'disrupts' and side in ('downstream', 'upstream') and (cds.partial or cds.pseudo or distance < 0):
        return 'disrupted'
    return 'none'


def inside_is(cds: Cds, bed: dict, exclude: str) -> str:
    """The name of another truth IS that contains the CDS, or '' when there is none."""
    for (replicon, start, end), (name, _) in bed.items():
        if name != exclude and replicon == cds.replicon and cds.start >= start and cds.end <= end:
            return name
    return ''


def build(curated: list[dict], genes: dict[str, Cds], bed: dict) -> tuple[list[dict], list[dict]]:
    truth, review = [], []
    genes_inside = {}
    for (replicon, start, end) in bed:
        genes_inside[(replicon, start, end)] = sorted(
            ((tag, cds) for tag, cds in genes.items()
             if cds.replicon == replicon and cds.start >= start and cds.end <= end), key=lambda tc: tc[1].start)
    for row in curated:
        replicon, start, end = strip_refseq(row['Chr']), int(row['Start']), int(row['End'])
        is_id, is_strand = bed.get((replicon, start, end), (None, None))
        row_effect, note = row.get('Effect', ''), (row.get('Note') or '').strip()

        def flag(reason, **extra):
            review.append({'is_id': is_id or row['IS_name'], 'replicon': replicon, 'is_start': start,
                           'is_end': end, 'reason': reason, **extra})

        if is_id is None:
            flag('IS coordinates not in the truth BED')
            continue
        if row_effect.strip().lower() not in SCORED_EFFECTS:
            flag(f'curated effect is "{row_effect.strip()}"')

        disrupted_found, seen = False, set()
        for tag, cds in genes_inside.get((replicon, start, end), []):
            seen.add((is_id, tag))
            truth.append({'is_id': is_id, 'replicon': replicon, 'is_start': start, 'is_end': end,
                          'is_strand': is_strand, 'side': 'inside', 'locus_tag': tag, 'gene': cds.gene,
                          'product': cds.product, 'cds_start': cds.start, 'cds_end': cds.end,
                          'cds_strand': cds.strand, 'partial': cds.partial, 'pseudo': cds.pseudo,
                          'distance_bp': 0, 'effect': 'none', 'curated_effect': row_effect.strip(),
                          'inside_is': '', 'note': note})
        for curated_side in CURATED_SIDES:
            cell = row.get(curated_side.capitalize(), '')
            for tag, curated_partial, curated_distance in parse_cell(cell):
                context = {'curated_side': curated_side, 'locus_tag': tag, 'curated_text': cell.strip()}
                cds = genes.get(tag)
                if cds is None:
                    flag('locus tag not in the RefSeq annotation (truncated or mistyped?)', **context)
                    continue
                if cds.replicon != replicon:
                    flag(f'locus tag is on {cds.replicon}, not {replicon}', **context)
                    continue
                side, distance = relate(start, end, is_strand, cds)
                context.update(side=side, distance_bp=distance, curated_distance=curated_distance)
                if abs(distance) > MAX_REACH:
                    flag(f'CDS is {distance} bp from the IS', **context)
                    continue
                if side != curated_side:
                    flag(f'curated as {curated_side}, coordinates say {side}', **context)
                if curated_distance is not None and (abs(distance - curated_distance) > DISTANCE_TOLERANCE
                                                     or ((distance < 0) != (curated_distance < 0)
                                                         and abs(distance - curated_distance) > 10)):
                    flag(f'distance {distance} bp, curated {curated_distance} bp', **context)
                if curated_partial is not None and curated_partial != cds.partial and not cds.pseudo:
                    flag(f'annotation partial={cds.partial}, curated partial={curated_partial}', **context)

                if (is_id, tag) in seen:
                    continue  # the same CDS curated on both sides; its coordinates decide which
                seen.add((is_id, tag))
                effect = relation_effect(row_effect, side, cds, distance)
                disrupted_found |= effect == 'disrupted'
                truth.append({'is_id': is_id, 'replicon': replicon, 'is_start': start, 'is_end': end,
                              'is_strand': is_strand, 'side': side, 'locus_tag': tag, 'gene': cds.gene,
                              'product': cds.product, 'cds_start': cds.start, 'cds_end': cds.end,
                              'cds_strand': cds.strand, 'partial': cds.partial, 'pseudo': cds.pseudo,
                              'distance_bp': distance, 'effect': effect, 'curated_effect': row_effect.strip(),
                              'inside_is': inside_is(cds, bed, exclude=is_id), 'note': note})
        if row_effect.strip().lower() == 'disrupts' and not disrupted_found:
            flag('curated as Disrupts, but no flanking CDS is partial, pseudo or overlapping')
    return truth, review


def write_tsv(path: Path, rows: list[dict], fields: list[str]):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('curated', type=Path)
    parser.add_argument('gff', type=Path)
    parser.add_argument('bed', type=Path)
    parser.add_argument('-o', '--out', type=Path, default=Path('data/ec958_IS_cds_relations.tsv'))
    parser.add_argument('-r', '--review', type=Path, default=Path('data/ec958_IS_cds_relations.review.tsv'))
    args = parser.parse_args(argv)

    truth, review = build(read_curated(args.curated), read_gff(args.gff), read_bed(args.bed))
    write_tsv(args.out, truth, TRUTH_FIELDS)
    write_tsv(args.review, review, ['is_id', 'replicon', 'is_start', 'is_end', 'reason', 'curated_side',
                                    'locus_tag', 'side', 'distance_bp', 'curated_distance', 'curated_text'])
    effects = {e: sum(r['effect'] == e for r in truth) for e in ('disrupted', 'none', 'uncertain')}
    print(f"{len(truth)} relations written to {args.out} ({effects}); "
          f"{len(review)} items to review in {args.review}", file=sys.stderr)


if __name__ == '__main__':
    main()
