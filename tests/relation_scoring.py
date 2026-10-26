"""
Scores eris's relations against the EC958 relation truth (built by scripts/build_relation_truth.py).

eris genes are pyfgs predictions on the assembly, so each one is first placed on the reference genome; a gene then
matches a RefSeq CDS when it lies on the same replicon and strand and overlaps it by at least half of the shorter
of the two. A truth relation is keyed by (IS family, side, locus tag) and is found when any eris locus of that
family reports a matching flank on that side. See docs/adr/0002-relation-benchmark-truth.md.
"""
import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import mappy
from Bio import SeqIO
from Bio.Seq import reverse_complement

from eris.constants import Context, Effect

SIDES = ('upstream', 'downstream')
# A gene crossing an end of the IS is reported as a passenger, but it is a flank on the side it sticks out of
OVERLAP_SIDES = {Context.OVERLAPPING_START: 'upstream', Context.OVERLAPPING_END: 'downstream'}
MIN_OVERLAP = 0.5  # of the shorter of the eris gene and the RefSeq CDS
DISRUPTING = Effect.DISRUPTED | Effect.TRUNCATED  # a truncation is a kind of disruption
MIN_IDENTITY = 0.99  # over the aligned part, for a matched flank's sequence to count as correct
MIN_CDS_COVERAGE = 0.8  # of the RefSeq CDS; waived when it is partial or pseudo
NEAR = 150  # bp; pyfgs start codons and eris's IS ends differ from RefSeq by a few to tens of bp, so distances are banded


@dataclass(frozen=True)
class TruthRelation:
    family: str
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

    @property
    def key(self) -> tuple[str, str, str]:
        return self.family, self.side, self.locus_tag


@dataclass(frozen=True)
class Placement:
    """Where an eris gene lies on the reference: `strand` is the strand its coding sequence reads on."""
    replicon: str
    start: int
    end: int
    strand: str
    identity: float  # over the aligned part


def read_truth(path: Path, families: dict[str, str], replicons: set[str] | None = None) -> list[TruthRelation]:
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
        name = r['is_id'].split('_')[0]
        scored = not r['inside_is'] and r['effect'] in ('disrupted', 'none')
        truths.append(TruthRelation(
            family=families.get(name, name), side=r['side'], locus_tag=r['locus_tag'], replicon=r['replicon'],
            start=int(r['cds_start']), end=int(r['cds_end']), strand=r['cds_strand'], partial=r['partial'] == 'True',
            pseudo=r['pseudo'] == 'True', distance_bp=int(r['distance_bp']), effect=r['effect'] if scored else None))
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
class Flank:
    """One flank reported by eris, read from a locus."""
    family: str
    side: str
    relation: object  # eris FeatureRelation


@dataclass
class RelationScore:
    total: dict[str, int] = field(default_factory=dict)  # truth relations per side
    found: set[tuple[str, str, str]] = field(default_factory=set)
    misses: set[tuple[str, str, str]] = field(default_factory=set)
    budget_misses: set[tuple[str, str, str]] = field(default_factory=set)  # misses found with a larger hops budget
    nucleotide_mismatches: set[tuple[str, str, str]] = field(default_factory=set)  # found, but the sequence is off
    found_any_side: set[tuple[str, str, str]] = field(default_factory=set)  # the gene was found, maybe on the other side
    first_flanks: int = 0  # eris's first flank per side per branch
    true_first_flanks: int = 0  # ... that match a truth relation
    bands: list[tuple[str, str]] = field(default_factory=list)  # (truth, eris) distance band of each found relation
    distance_errors: list[int] = field(default_factory=list)
    # Precision counts eris's disrupted first flanks; recall counts the truth's disrupted relations
    disrupted: dict[str, int] = field(default_factory=lambda: {'called': 0, 'called_true': 0, 'truth': 0, 'truth_found': 0})
    upregulated: list[tuple[str, str, str]] = field(default_factory=list)  # (family, side, eris gene); not scored
    unplaceable: list[str] = field(default_factory=list)  # reported flank genes with no place on the reference

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
        d = self.disrupted
        return d['called_true'] / d['called'] if d['called'] else 0.0

    @property
    def disrupted_recall(self) -> float:
        d = self.disrupted
        return d['truth_found'] / d['truth'] if d['truth'] else 0.0

    @property
    def band_agreement(self) -> float:
        return sum(t == e for t, e in self.bands) / len(self.bands) if self.bands else 0.0

    @property
    def median_distance_error(self) -> float:
        return median(self.distance_errors) if self.distance_errors else 0.0

    @property
    def max_distance_error(self) -> int:
        return max(self.distance_errors, default=0)


def locus_family(locus, families: dict[str, str]) -> str:
    name = locus.targets[0].id.split('_')[0]
    return families.get(name, name)


def read_flanks(locus, families: dict[str, str]) -> list[Flank]:
    """Every flank of a locus, nearest first on each side, including passengers that cross an end of the IS."""
    family = locus_family(locus, families)
    flanks = [Flank(family, OVERLAP_SIDES[r.spatial], r) for r in locus.passengers if r.spatial in OVERLAP_SIDES]
    flanks += [Flank(family, 'upstream', r) for r in locus.upstream_flanks]
    flanks += [Flank(family, 'downstream', r) for r in locus.downstream_flanks]
    return flanks


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
        nearest_hop = min(f.relation.topological_dist for f in on_side)
        contigs = set()
        for f in on_side:
            contig = f.relation.feature.segments[0].contig
            if f.relation.topological_dist == nearest_hop and contig not in contigs:
                contigs.add(contig)
                firsts.append(f)
            if nearest_hop == 0:
                break
    return firsts


def matches(placement: Placement, truth: TruthRelation) -> bool:
    if placement.replicon != truth.replicon or placement.strand != truth.strand:
        return False
    overlap = min(placement.end, truth.end) - max(placement.start, truth.start)
    return overlap >= MIN_OVERLAP * min(placement.end - placement.start, truth.end - truth.start)


def sequence_agrees(placement: Placement, truth: TruthRelation) -> bool:
    """Whether a matched eris gene has the RefSeq CDS's sequence; boundaries of partial or pseudo CDS are not checked."""
    covered = (min(placement.end, truth.end) - max(placement.start, truth.start)) / (truth.end - truth.start)
    return placement.identity >= MIN_IDENTITY and (covered >= MIN_CDS_COVERAGE or truth.partial or truth.pseudo)


def matching_flanks(truth: TruthRelation, flanks: list[Flank], place: Callable[[str], list[Placement]]) -> list[Flank]:
    """Flanks of the truth relation's family, on either side, whose gene matches its CDS."""
    return [f for f in flanks if f.family == truth.family and any(matches(p, truth) for p in place(f.relation.feature.id))]


def score_relations(truths: list[TruthRelation], loci: list, families: dict[str, str],
                    place: Callable[[str], list[Placement]], reach_loci: list | None = None) -> RelationScore:
    """
    Scores eris loci against the truth. `place` returns the reference placements of an eris gene by its id, or an
    empty list when the gene cannot be placed. `reach_loci` are the same assembly's loci from a run with a larger
    hops budget: a miss they find is put down to the budget rather than to a wrong gene.
    """
    per_locus = [read_flanks(locus, families) for locus in loci]
    flanks = [f for locus_flanks in per_locus for f in locus_flanks]
    score = RelationScore(total={side: sum(t.side == side for t in truths) for side in SIDES})
    score.unplaceable = sorted({f.relation.feature.id for f in flanks if not place(f.relation.feature.id)})
    score.upregulated = [(f.family, f.side, f.relation.feature.id) for f in flanks
                         if f.relation.effect & Effect.UPREGULATED]
    for f in (f for locus_flanks in per_locus for f in first_flanks(locus_flanks)):
        placements = place(f.relation.feature.id)
        true = [t for t in truths if t.family == f.family and t.side == f.side and any(matches(p, t) for p in placements)]
        score.first_flanks += 1
        score.true_first_flanks += bool(true)
        # A flank whose effect is not scored is left out
        if f.relation.effect & DISRUPTING and not any(t.effect is None for t in true):
            score.disrupted['called'] += 1
            score.disrupted['called_true'] += any(t.effect == 'disrupted' for t in true)
    for t in truths:
        hits = matching_flanks(t, flanks, place)
        if hits:
            score.found_any_side.add(t.key)
        on_side = [f.relation for f in hits if f.side == t.side]
        nearest = min(on_side, key=lambda r: (r.topological_dist, r.distance_bp), default=None)
        if t.effect == 'disrupted':
            score.disrupted['truth'] += 1
            score.disrupted['truth_found'] += bool(nearest and nearest.effect & DISRUPTING)
        if nearest is None:
            score.misses.add(t.key)
            continue
        score.found.add(t.key)
        score.bands.append((band(t.distance_bp), band(nearest.distance_bp, nearest.spatial in OVERLAP_SIDES)))
        # eris reports an overlap as distance 0
        score.distance_errors.append(abs(nearest.distance_bp - max(0, t.distance_bp)))
        if not any(sequence_agrees(p, t) for p in place(nearest.feature.id) if matches(p, t)):
            score.nucleotide_mismatches.add(t.key)

    if reach_loci is not None:
        reach_flanks = [f for locus in reach_loci for f in read_flanks(locus, families)]
        score.budget_misses = {t.key for t in truths if t.key in score.misses
                               and any(f.side == t.side for f in matching_flanks(t, reach_flanks, place))}
    return score


def report(score: RelationScore) -> str:
    """A readable summary of a score, for the test log."""
    found = {side: sum(k[1] == side for k in score.found) for side in SIDES}
    d = score.disrupted
    lines = [
        *(f"recall {side}: {found[side]}/{score.total[side]} = {score.recall[side]:.3f}" for side in SIDES),
        f"recall pooled over sides: {len(score.found_any_side)}/{sum(score.total.values())} = {score.pooled_recall:.3f}",
        f"precision of first flanks: {score.true_first_flanks}/{score.first_flanks} = {score.precision:.3f}",
        f"distance band agreement: {score.band_agreement:.3f} (median error {score.median_distance_error} bp, "
        f"max {score.max_distance_error} bp)",
        f"disrupted precision: {d['called_true']}/{d['called']} = {score.disrupted_precision:.3f}",
        f"disrupted recall: {d['truth_found']}/{d['truth']} = {score.disrupted_recall:.3f}",
        f"misses: {len(score.misses)}, of which {len(score.budget_misses)} within reach of a larger hops budget",
        *(f"  {'budget' if k in score.budget_misses else 'gene  '} {k}" for k in sorted(score.misses)),
        f"nucleotide mismatches: {sorted(score.nucleotide_mismatches)}",
        f"UPREGULATED calls (not scored): {score.upregulated}",
        f"unplaceable flank genes: {score.unplaceable}",
    ]
    return "\n".join(lines)
