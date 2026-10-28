"""
Scores eris's relations against a curated relation truth (for EC958, built by scripts/build_relation_truth.py).

eris genes are pyfgs predictions on the assembly, so each one is first placed on the reference genome; a gene then
matches a RefSeq CDS when it lies on the same replicon and strand and overlaps it by at least half of the shorter
of the two. A truth relation is keyed by (IS group, side, locus tag) and is found when any eris locus of that group
reports a matching flank on that side. The IS group comes from a key function: the IS family for the EC958 ratchet,
or the ISfinder synonym group for `eris-validate`. See docs/adr/0002-relation-benchmark-truth.md.
"""
import csv
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import reduce
from operator import or_
from pathlib import Path
from statistics import median

import mappy
from Bio import SeqIO
from Bio.Seq import reverse_complement

from eris.constants import Context, Effect

SIDES = ('upstream', 'downstream')
# A gene crossing an end of the IS is reported as a passenger, but it is a flank on the side it sticks out of
OVERLAP_SIDES = {Context.OVERLAPPING_START: 'upstream', Context.OVERLAPPING_END: 'downstream'}
FLANK_SIDES = {Context.UPSTREAM: 'upstream', Context.DOWNSTREAM: 'downstream', **OVERLAP_SIDES}
MIN_OVERLAP = 0.5  # of the shorter of the eris gene and the RefSeq CDS
DISRUPTING = Effect.DISRUPTED | Effect.TRUNCATED  # a truncation is a kind of disruption
MIN_IDENTITY = 0.99  # over the aligned part, for a matched flank's sequence to count as correct
MIN_CDS_COVERAGE = 0.8  # of the RefSeq CDS; waived when it is partial or pseudo
RelationKey = tuple[str, str, str]  # (IS group, side, locus tag)
NEAR = 150  # bp; pyfgs start codons and eris's IS ends differ from RefSeq by a few to tens of bp, so distances are banded
BANDS = ('overlapping', 'near', 'far')


@dataclass(frozen=True)
class TruthRelation:
    group: str
    side: str
    locus_tag: str
    replicon: str
    start: int  # 0-based, half-open
    end: int
    strand: str
    partial: bool
    pseudo: bool
    distance_bp: int  # negative when the CDS overlaps the IS
    effect: str | None  # 'disrupted' or 'none'; None when the effect is not scored
    is_id: str = ''  # the IS copy, as named in the truth

    @property
    def key(self) -> RelationKey:
        return self.group, self.side, self.locus_tag


@dataclass(frozen=True)
class Placement:
    """Where an eris gene lies on the reference: `strand` is the strand its coding sequence reads on."""
    replicon: str
    start: int
    end: int
    strand: str
    identity: float  # over the aligned part


@dataclass(frozen=True)
class Flank:
    """One flank reported by eris, read from a locus or from a row of the report."""
    group: str
    side: str
    gene_id: str
    contig: str  # the contig the gene was called on
    hops: int  # contigs between the IS and the gene; 0 on the IS's own contig
    distance_bp: int  # 0 when the gene overlaps the IS
    effect: Effect
    overlapping: bool  # crosses an end of the IS
    locus_id: str = ''


def read_truth(path: Path, group_of: Callable[[str], str], replicons: set[str] | None = None) -> list[TruthRelation]:
    """
    Reads the flank relations of the truth TSV, optionally only those on the given replicons. The effect of a flank
    that lies inside another IS, or whose curated effect is uncertain, is not scored.
    """
    with open(path) as handle:
        rows = [r for r in csv.DictReader(handle, delimiter='\t') if r['side'] in SIDES]
    truths = []
    for r in rows:
        if replicons is not None and r['replicon'] not in replicons:
            continue
        scored = not r['inside_is'] and r['effect'] in ('disrupted', 'none')
        truths.append(TruthRelation(
            group=group_of(r['is_id']), side=r['side'], locus_tag=r['locus_tag'], replicon=r['replicon'],
            start=int(r['cds_start']), end=int(r['cds_end']), strand=r['cds_strand'], partial=r['partial'] == 'True',
            pseudo=r['pseudo'] == 'True', distance_bp=int(r['distance_bp']), effect=r['effect'] if scored else None,
            is_id=r['is_id']))
    return truths


class GenePlacer:
    """
    Places eris genes on the reference genome by aligning their coding sequence with minimap2. A gene that aligns
    equally well to several places (e.g. a repeated gene) gets all of them; one too short to align is looked up by
    exact search.
    """

    def __init__(self, reference: Path):
        self._aligner = mappy.Aligner(str(reference), preset='asm5')
        self._replicons = {r.id: str(r.seq).upper() for r in SeqIO.parse(reference, 'fasta')}

    def place(self, sequence: bytes) -> list[Placement]:
        seq = sequence.decode().upper()
        hits = list(self._aligner.map(seq))
        if hits:
            best = max(h.mlen for h in hits)
            return [Placement(h.ctg, h.r_st, h.r_en, '+' if h.strand == 1 else '-', h.mlen / h.blen)
                    for h in hits if h.mlen == best]
        return self._search(seq)

    def _search(self, seq: str) -> list[Placement]:
        placements = []
        for name, replicon in self._replicons.items():
            for strand, query in (('+', seq), ('-', reverse_complement(seq))):
                start = replicon.find(query)
                while start != -1:
                    placements.append(Placement(name, start, start + len(seq), strand, 1.0))
                    start = replicon.find(query, start + 1)
        return placements


@dataclass(frozen=True)
class TruthRow:
    """How eris did on one truth relation."""
    truth: TruthRelation
    status: str  # 'found', 'wrong_side' (the gene was only reported on the other side) or 'missed'
    flank: Flank | None  # the nearest matching flank on the truth's side, or on the other side when wrong_side
    sequence_ok: bool | None  # None unless found
    miss_cause: str = ''  # 'budget', 'gene' or 'unknown' for a relation not found on its side


@dataclass(frozen=True)
class FlankRow:
    """How one of eris's first flanks matched the truth: the unit of precision."""
    flank: Flank
    status: str  # 'true', 'wrong_side' (matches a truth relation on the other side only) or 'no_truth'
    replicon: str  # of the matched truth relation, else of the gene's first placement; '' when unplaceable
    truth_effect: str  # of the matched truth relations: 'disrupted', 'none', 'unscored', or '' when none matched


@dataclass
class RelationScore:
    total: dict[str, int] = field(default_factory=dict)  # truth relations per side
    found: set[RelationKey] = field(default_factory=set)
    misses: set[RelationKey] = field(default_factory=set)
    budget_misses: set[RelationKey] = field(default_factory=set)  # misses found with a larger hops budget
    nucleotide_mismatches: set[RelationKey] = field(default_factory=set)  # found, but the sequence is off
    found_any_side: set[RelationKey] = field(default_factory=set)  # the gene was found, maybe on the other side
    first_flanks: int = 0  # eris's first flank per side per branch
    true_first_flanks: int = 0  # ... that match a truth relation
    bands: list[tuple[str, str]] = field(default_factory=list)  # (truth, eris) distance band of each found relation
    distance_errors: list[int] = field(default_factory=list)
    # Disrupted precision is over eris's first flanks called DISRUPTED or TRUNCATED, like `precision`; a disrupted
    # call further along a branch is not counted. Recall is over the truth's disrupted relations.
    disrupted_called: int = 0
    disrupted_called_true: int = 0
    disrupted_truth: int = 0
    disrupted_truth_found: int = 0
    upregulated: list[tuple[str, str, str]] = field(default_factory=list)  # (group, side, eris gene); not scored
    unplaceable: list[str] = field(default_factory=list)  # reported flank genes with no place on the reference
    truth_rows: list[TruthRow] = field(default_factory=list)
    flank_rows: list[FlankRow] = field(default_factory=list)

    @property
    def miss_causes(self) -> dict[RelationKey, str]:
        """Each miss with its cause: 'budget' if a larger hops budget finds it, else 'gene'."""
        return {k: 'budget' if k in self.budget_misses else 'gene' for k in self.misses}

    @property
    def recall(self) -> dict[str, float]:
        return {side: sum(k[1] == side for k in self.found) / n if n else 0.0 for side, n in self.total.items()}

    @property
    def pooled_recall(self) -> float:
        """Separates finding the gene from reading the IS strand right."""
        n = sum(self.total.values())
        return len(self.found_any_side) / n if n else 0.0

    @property
    def precision(self) -> float:
        return self.true_first_flanks / self.first_flanks if self.first_flanks else 0.0

    @property
    def disrupted_precision(self) -> float:
        return self.disrupted_called_true / self.disrupted_called if self.disrupted_called else 0.0

    @property
    def disrupted_recall(self) -> float:
        return self.disrupted_truth_found / self.disrupted_truth if self.disrupted_truth else 0.0

    @property
    def band_agreement(self) -> float:
        return sum(t == e for t, e in self.bands) / len(self.bands) if self.bands else 0.0

    @property
    def median_distance_error(self) -> float:
        return median(self.distance_errors) if self.distance_errors else 0.0

    @property
    def max_distance_error(self) -> int:
        return max(self.distance_errors, default=0)


def family_of(name: str, families: dict[str, str]) -> str:
    """The IS family of a target or truth IS name, e.g. 'IS600_IS3_IS3' or 'IS600_2' -> 'IS3'."""
    name = name.split('_')[0]
    return families.get(name, name)


def read_flanks(locus, group_of: Callable[[str], str]) -> list[Flank]:
    """Every flank of a locus, nearest first on each side, including passengers that cross an end of the IS."""
    group = group_of(locus.targets[0].id)
    relations = [r for r in locus.passengers if r.spatial in OVERLAP_SIDES] + locus.upstream_flanks + \
        locus.downstream_flanks
    return [Flank(group, FLANK_SIDES[r.spatial], r.feature.id, r.feature.segments[0].contig, r.topological_dist,
                  r.distance_bp, r.effect, r.spatial in OVERLAP_SIDES, locus.id) for r in relations]


def parse_effect(name: str) -> Effect:
    """An Effect from its name in the report, e.g. 'DISRUPTED' or 'TRUNCATED|DISRUPTED'."""
    return reduce(or_, (Effect[n] for n in name.split('|')))


def read_report_flanks(path: Path, group_of: Callable[[str], str]) -> list[list[Flank]]:
    """
    The flanks of each locus in an eris `_report.tsv`, nearest first on each side, as `read_flanks` reads them from
    loci. A gene id is '<contig>_<start>_<end>', so the contig is everything before the last two fields.
    """
    loci: dict[str, list[Flank]] = {}
    with open(path) as handle:
        for r in csv.DictReader(handle, delimiter='\t'):
            flanks = loci.setdefault(r['locus_id'], [])
            if (context := Context.__members__.get(r['context'])) not in FLANK_SIDES:
                continue
            flanks.append(Flank(
                group_of(r['target'].split(',')[0]), FLANK_SIDES[context], r['gene_id'],
                r['gene_id'].rsplit('_', 2)[0], int(r['topo_hops']), int(r['dist_bp']), parse_effect(r['effect']),
                context in OVERLAP_SIDES, r['locus_id']))
    return list(loci.values())


# Files written by pyfgs's FnaWriter (eris before records were named by gene_id) carry '<contig>:<start>-<end>',
# 1-based, in each header's description
_PYFGS_LOCATION = re.compile(r'(\S+):(\d+)-(\d+)')


def read_cds_sequences(path: Path) -> dict[str, bytes]:
    """
    The coding sequence of every gene in an eris `_cds.fna`, by the gene id the report uses ('<contig>_<start>_<end>',
    0-based). eris names each record by that id; an older file written by pyfgs is read from its 1-based location.
    """
    sequences = {}
    for record in SeqIO.parse(path, 'fasta'):
        if m := _PYFGS_LOCATION.search(record.description):
            sequences[f"{m.group(1)}_{int(m.group(2)) - 1}_{m.group(3)}"] = bytes(record.seq)
        else:
            sequences[record.id] = bytes(record.seq)
    return sequences


def band(distance: int, overlapping: bool = False) -> str:
    if overlapping or distance < 0:
        return 'overlapping'
    return 'near' if distance <= NEAR else 'far'


def first_flanks(flanks: list[Flank]) -> list[Flank]:
    """
    The nearest flank of each branch on each side of one locus. A flank on the IS's own contig is the only branch on
    its side. Otherwise each contig at the fewest hops that carries a gene starts a branch; flanks further along are
    left out, since eris does not report which branch a deeper contig continues.
    """
    firsts = []
    for side in SIDES:
        on_side = [f for f in flanks if f.side == side]
        if not on_side:
            continue
        nearest_hop = min(f.hops for f in on_side)
        contigs = set()
        for f in on_side:
            if f.hops == nearest_hop and f.contig not in contigs:
                contigs.add(f.contig)
                firsts.append(f)
            if nearest_hop == 0:
                break
    return firsts


def overlap(placement: Placement, truth: TruthRelation) -> int:
    return min(placement.end, truth.end) - max(placement.start, truth.start)


def matches(placement: Placement, truth: TruthRelation) -> bool:
    if placement.replicon != truth.replicon or placement.strand != truth.strand:
        return False
    return overlap(placement, truth) >= MIN_OVERLAP * min(placement.end - placement.start, truth.end - truth.start)


def sequence_agrees(placement: Placement, truth: TruthRelation) -> bool:
    """Whether a matched eris gene has the RefSeq CDS's sequence; boundaries of partial or pseudo CDS are not checked."""
    covered = overlap(placement, truth) / (truth.end - truth.start)
    return placement.identity >= MIN_IDENTITY and (covered >= MIN_CDS_COVERAGE or truth.partial or truth.pseudo)


def matching_flanks(truth: TruthRelation, flanks: list[Flank], place: Callable[[str], list[Placement]]) -> list[Flank]:
    """Flanks of the truth relation's group, on either side, whose gene matches its CDS."""
    return [f for f in flanks if f.group == truth.group and any(matches(p, truth) for p in place(f.gene_id))]


def _nearest(flanks: Iterable[Flank]) -> Flank | None:
    return min(flanks, key=lambda f: (f.hops, f.distance_bp), default=None)


def _truth_effect(truths: list[TruthRelation]) -> str:
    if not truths:
        return ''
    if any(t.effect is None for t in truths):
        return 'unscored'
    return 'disrupted' if any(t.effect == 'disrupted' for t in truths) else 'none'


def _miss_cause(truth: TruthRelation, reach_flanks: list[Flank] | None, place: Callable[[str], list[Placement]]) -> str:
    """'budget' if the larger hops budget finds the missed relation, 'gene' if not, 'unknown' without such a run."""
    if reach_flanks is None:
        return 'unknown'
    return 'budget' if any(f.side == truth.side for f in matching_flanks(truth, reach_flanks, place)) else 'gene'


def score_flanks(truths: list[TruthRelation], per_locus: list[list[Flank]], place: Callable[[str], list[Placement]],
                 reach: list[list[Flank]] | None = None) -> RelationScore:
    """
    Scores eris's flanks, grouped by locus, against the truth. `place` returns the reference placements of an eris
    gene by its id, or an empty list when the gene cannot be placed. `reach` are the same assembly's flanks from a run
    with a larger hops budget: a miss they find is put down to the budget rather than to a wrong gene.
    """
    # eris cannot tell copies of one IS group apart, so two copies flanked by the same CDS (e.g. tandem IS1A copies)
    # are one relation: the nearer copy's
    nearest_copy = {}
    for t in sorted(truths, key=lambda t: abs(t.distance_bp)):
        nearest_copy.setdefault(t.key, t)
    truths = [t for t in truths if nearest_copy[t.key] is t]
    flanks = [f for locus_flanks in per_locus for f in locus_flanks]
    score = RelationScore(total={side: sum(t.side == side for t in truths) for side in SIDES})
    score.unplaceable = sorted({f.gene_id for f in flanks if not place(f.gene_id)})
    score.upregulated = [(f.group, f.side, f.gene_id) for f in flanks if f.effect & Effect.UPREGULATED]
    for f in (f for locus_flanks in per_locus for f in first_flanks(locus_flanks)):
        placements = place(f.gene_id)
        any_side = [t for t in truths if t.group == f.group and any(matches(p, t) for p in placements)]
        matched = [t for t in any_side if t.side == f.side]
        score.first_flanks += 1
        score.true_first_flanks += bool(matched)
        # A flank whose effect is not scored is left out
        if f.effect & DISRUPTING and not any(t.effect is None for t in matched):
            score.disrupted_called += 1
            score.disrupted_called_true += any(t.effect == 'disrupted' for t in matched)
        status = 'true' if matched else 'wrong_side' if any_side else 'no_truth'
        replicon = (matched or any_side)[0].replicon if any_side else placements[0].replicon if placements else ''
        score.flank_rows.append(FlankRow(f, status, replicon, _truth_effect(matched)))

    reach_flanks = [f for locus_flanks in reach for f in locus_flanks] if reach is not None else None
    for t in truths:
        matching = matching_flanks(t, flanks, place)
        if matching:
            score.found_any_side.add(t.key)
        nearest = _nearest(f for f in matching if f.side == t.side)
        if t.effect == 'disrupted':
            score.disrupted_truth += 1
            score.disrupted_truth_found += bool(nearest and nearest.effect & DISRUPTING)
        if nearest is None:
            score.misses.add(t.key)
            if (cause := _miss_cause(t, reach_flanks, place)) == 'budget':
                score.budget_misses.add(t.key)
            other = _nearest(matching)
            score.truth_rows.append(TruthRow(t, 'wrong_side' if other else 'missed', other, None, cause))
            continue
        score.found.add(t.key)
        score.bands.append((band(t.distance_bp), band(nearest.distance_bp, nearest.overlapping)))
        # eris reports an overlap as distance 0
        score.distance_errors.append(abs(nearest.distance_bp - max(0, t.distance_bp)))
        sequence_ok = any(sequence_agrees(p, t) for p in place(nearest.gene_id) if matches(p, t))
        if not sequence_ok:
            score.nucleotide_mismatches.add(t.key)
        score.truth_rows.append(TruthRow(t, 'found', nearest, sequence_ok))
    return score


def score_relations(truths: list[TruthRelation], loci: list, group_of: Callable[[str], str],
                    place: Callable[[str], list[Placement]], reach_loci: list | None = None) -> RelationScore:
    """`score_flanks` on eris loci, as the pipeline returns them."""
    reach = [read_flanks(locus, group_of) for locus in reach_loci] if reach_loci is not None else None
    return score_flanks(truths, [read_flanks(locus, group_of) for locus in loci], place, reach)


def report(score: RelationScore) -> str:
    """A readable summary of a score, for the test log."""
    found = {side: sum(k[1] == side for k in score.found) for side in SIDES}
    lines = [
        *(f"recall {side}: {found[side]}/{score.total[side]} = {score.recall[side]:.3f}" for side in SIDES),
        f"recall pooled over sides: {len(score.found_any_side)}/{sum(score.total.values())} = {score.pooled_recall:.3f}",
        f"precision of first flanks: {score.true_first_flanks}/{score.first_flanks} = {score.precision:.3f}",
        f"distance band agreement: {score.band_agreement:.3f} (median error {score.median_distance_error} bp, "
        f"max {score.max_distance_error} bp)",
        f"disrupted precision: {score.disrupted_called_true}/{score.disrupted_called} = {score.disrupted_precision:.3f}",
        f"disrupted recall: {score.disrupted_truth_found}/{score.disrupted_truth} = {score.disrupted_recall:.3f}",
        f"misses: {len(score.misses)}, of which {len(score.budget_misses)} within reach of a larger hops budget",
        *(f"  {cause:6} {k}" for k, cause in sorted(score.miss_causes.items())),
        f"nucleotide mismatches: {sorted(score.nucleotide_mismatches)}",
        f"UPREGULATED calls (not scored): {score.upregulated}",
        f"unplaceable flank genes: {score.unplaceable}",
    ]
    return "\n".join(lines)
