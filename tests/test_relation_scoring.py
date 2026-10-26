import random
from types import SimpleNamespace

from Bio.Seq import reverse_complement

from eris.constants import Context, Effect
from relation_scoring import GenePlacer, Placement, TruthRelation, score_relations

FAMILIES = {"ISA": "famA", "ISB": "famB"}


def relation(gene_id, spatial, dist=0, topo=0, effect=Effect.NONE, contig="ctg"):
    feature = SimpleNamespace(id=gene_id, segments=[SimpleNamespace(contig=contig)])
    return SimpleNamespace(feature=feature, spatial=spatial, distance_bp=dist, topological_dist=topo, effect=effect)


def locus(name, passengers=(), up=(), down=()):
    return SimpleNamespace(id=f"locus_{name}", targets=[SimpleNamespace(id=f"{name}_fam_grp")],
                           passengers=list(passengers), upstream_flanks=list(up), downstream_flanks=list(down))


def truth(side, tag, start, end, family="famA", strand="+", distance=10, effect="none", partial=False,
          scored=True):
    return TruthRelation(family=family, side=side, locus_tag=tag, replicon="chr", start=start, end=end,
                         strand=strand, partial=partial, pseudo=False, distance_bp=distance,
                         effect=effect if scored else None)


def placed(**genes):
    """gene id -> (start, end[, strand[, identity]]) on 'chr'; genes left out are unplaceable."""
    table = {g: [Placement("chr", *(v + ("+", 1.0)[len(v) - 2:]))] for g, v in genes.items()}
    return lambda gene_id: table.get(gene_id, [])


def test_relation_is_found_by_family_side_and_overlapping_gene():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000)]
    loci = [locus("ISA", up=[relation("g1", Context.UPSTREAM, 30)], down=[relation("g2", Context.DOWNSTREAM, 40)])]
    # g1 covers half of T1; g2 lies on the other strand of T2
    score = score_relations(truths, loci, FAMILIES, placed(g1=(1500, 2600), g2=(5000, 6000, "-")))

    assert score.found == {("famA", "upstream", "T1")}
    assert score.recall == {"upstream": 1.0, "downstream": 0.0}


def test_gene_crossing_an_is_end_is_a_flank_on_the_side_it_sticks_out_of():
    truths = [truth("upstream", "T1", 1000, 2000, distance=-20), truth("downstream", "T2", 5000, 6000, distance=-5)]
    loci = [locus("ISA", passengers=[relation("g1", Context.OVERLAPPING_START), relation("g2", Context.OVERLAPPING_END),
                                     relation("tnp", Context.INSIDE)])]
    score = score_relations(truths, loci, FAMILIES, placed(g1=(1000, 2000), g2=(5000, 6000), tnp=(2000, 5000)))

    assert score.recall == {"upstream": 1.0, "downstream": 1.0}


def test_pooled_recall_forgives_a_flank_found_on_the_wrong_side():
    # The gene was found, but eris read the IS on the other strand
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000)]
    loci = [locus("ISA", down=[relation("g1", Context.DOWNSTREAM)])]
    score = score_relations(truths, loci, FAMILIES, placed(g1=(1000, 2000)))

    assert score.recall == {"upstream": 0.0, "downstream": 0.0}
    assert score.pooled_recall == 0.5


def test_flank_of_another_family_does_not_count():
    truths = [truth("upstream", "T1", 1000, 2000)]
    loci = [locus("ISB", up=[relation("g1", Context.UPSTREAM)])]
    assert score_relations(truths, loci, FAMILIES, placed(g1=(1000, 2000))).found == set()


def test_precision_scores_the_first_flank_of_each_branch():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T3", 7000, 8000)]
    local = locus("ISA", up=[relation("g1", Context.UPSTREAM, 10), relation("g2", Context.UPSTREAM, 900)])
    # A collapsed repeat: two flanking contigs one hop away, and a third further along one of them
    repeat = locus("ISA", down=[relation("g3", Context.DOWNSTREAM, 50, topo=1, contig="c1"),
                                relation("g4", Context.DOWNSTREAM, 800, topo=1, contig="c1"),
                                relation("g5", Context.DOWNSTREAM, 20, topo=1, contig="c2"),
                                relation("g6", Context.DOWNSTREAM, 3000, topo=2, contig="c3")])
    place = placed(g1=(1000, 2000), g2=(3000, 3500), g3=(7000, 8000), g4=(9000, 9500), g5=(20000, 21000),
                   g6=(30000, 31000))
    score = score_relations(truths, [local, repeat], FAMILIES, place)

    # First flanks: g1 and g3 are true, g5 is not; g2, g4 and g6 lie further along their branches
    assert score.precision == 2 / 3


def test_distance_band_agreement_on_found_relations():
    truths = [truth("upstream", "T1", 1000, 2000, distance=-12),  # overlapping, eris agrees
              truth("downstream", "T2", 5000, 6000, distance=140),  # near; eris says far
              truth("upstream", "T3", 9000, 9500, family="famB", distance=400)]  # far, eris agrees
    loci = [locus("ISA", passengers=[relation("g1", Context.OVERLAPPING_START)],
                  down=[relation("g2", Context.DOWNSTREAM, 160)]),
            locus("ISB", up=[relation("g3", Context.UPSTREAM, 380)])]
    score = score_relations(truths, loci, FAMILIES, placed(g1=(1000, 2000), g2=(5000, 6000), g3=(9000, 9500)))

    assert score.band_agreement == 2 / 3
    # Errors |eris - truth|, with overlaps read as 0 as eris reports them: 0, 20, 20
    assert (score.median_distance_error, score.max_distance_error) == (20, 20)


def test_disrupted_or_truncated_scores_against_truth_disrupted():
    truths = [truth("upstream", "T1", 1000, 2000, effect="disrupted"),
              truth("downstream", "T2", 5000, 6000, effect="disrupted"),
              truth("upstream", "T3", 9000, 9500, family="famB", effect="disrupted"),
              truth("downstream", "T4", 12000, 13000, family="famB", scored=False),  # inside another IS
              truth("upstream", "T5", 20000, 21000)]
    loci = [locus("ISA", passengers=[relation("g1", Context.OVERLAPPING_START, effect=Effect.DISRUPTED)],
                  down=[relation("g2", Context.DOWNSTREAM, 3, effect=Effect.TRUNCATED)]),
            locus("ISB", up=[relation("g3", Context.UPSTREAM, 10)],
                  passengers=[relation("g4", Context.OVERLAPPING_END, effect=Effect.DISRUPTED)]),
            locus("ISA", up=[relation("g5", Context.UPSTREAM, 0, effect=Effect.DISRUPTED)],
                  down=[relation("g6", Context.DOWNSTREAM, 40, effect=Effect.UPREGULATED)])]
    place = placed(g1=(1000, 2000), g2=(5000, 6000), g3=(9000, 9500), g4=(12000, 13000), g5=(20000, 21000),
                   g6=(30000, 31000))
    score = score_relations(truths, loci, FAMILIES, place)

    # g1 and g2 are right, g5 is a false call, g3 is missed; g4's effect is not scored
    assert (score.disrupted_precision, score.disrupted_recall) == (2 / 3, 2 / 3)
    assert score.upregulated == [("famA", "downstream", "g6")]


def test_misses_found_with_a_larger_hops_budget_are_told_apart():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000),
              truth("downstream", "T3", 9000, 9500)]
    loci = [locus("ISA", up=[relation("g1", Context.UPSTREAM)], down=[relation("short", Context.DOWNSTREAM)])]
    # With a larger budget the same locus also reaches g2, the true downstream flank; T3 stays out of reach
    reach = [locus("ISA", up=[relation("g1", Context.UPSTREAM)],
                   down=[relation("short", Context.DOWNSTREAM), relation("g2", Context.DOWNSTREAM, 900)])]
    score = score_relations(truths, loci, FAMILIES, placed(g1=(1000, 2000), g2=(5000, 6000)), reach_loci=reach)

    assert score.misses == {("famA", "downstream", "T2"), ("famA", "downstream", "T3")}
    assert score.budget_misses == {("famA", "downstream", "T2")}
    assert score.unplaceable == ["short"]


def test_nucleotide_check_needs_identity_and_coverage_of_complete_cds():
    truths = [truth("upstream", "T1", 1000, 2000), truth("downstream", "T2", 5000, 6000),
              truth("upstream", "T3", 9000, 10000, family="famB", partial=True),
              truth("downstream", "T4", 12000, 13000, family="famB")]
    loci = [locus("ISA", up=[relation("g1", Context.UPSTREAM)], down=[relation("g2", Context.DOWNSTREAM)]),
            locus("ISB", up=[relation("g3", Context.UPSTREAM)], down=[relation("g4", Context.DOWNSTREAM)])]
    place = placed(g1=(1000, 2000, "+", 0.98),  # too many mismatches
                   g2=(5300, 6000),  # covers 70% of the CDS
                   g3=(9400, 10000),  # 60%, but the RefSeq CDS is partial
                   g4=(12000, 13000, "+", 0.995))
    score = score_relations(truths, loci, FAMILIES, place)

    assert score.found == {t.key for t in truths}
    assert score.nucleotide_mismatches == {("famA", "upstream", "T1"), ("famA", "downstream", "T2")}


def test_genes_are_placed_by_alignment_or_exact_search(tmp_path):
    rng = random.Random(15)
    chrom = "".join(rng.choice("ACGT") for _ in range(20000))
    (tmp_path / "ref.fasta").write_text(f">chr\n{chrom}\n")
    placer = GenePlacer(tmp_path / "ref.fasta")

    assert placer.place(chrom[5000:5600].encode()) == [Placement("chr", 5000, 5600, "+", 1.0)]
    assert placer.place(reverse_complement(chrom[10000:10900]).encode()) == [Placement("chr", 10000, 10900, "-", 1.0)]
    # Too short for minimap2
    assert placer.place(reverse_complement(chrom[15000:15060]).encode()) == [Placement("chr", 15000, 15060, "-", 1.0)]
    assert placer.place(b"ACGT" * 15) == []
