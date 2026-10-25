import pytest
from types import SimpleNamespace
from pathlib import Path

from eris.io import GenomeAssembly, TargetDatabase
from eris.pipeline import Pipeline
from eris.constants import FeatureType

DATA_DIR = Path("data")

# The benchmark data is large and kept out of git; skip these tests when it is absent
REQUIRED_FILES = [
    "IS.mmi",
    "EC958_complete_genome.fasta", "complete_ec958_IS.reanotated.sorted.bed",
    "contigs_shovill_simulated.gfa", "ec958_IS.reannotated.sorted.bed",
]
needs_data = pytest.mark.skipif(
    not all((DATA_DIR / f).exists() for f in REQUIRED_FILES),
    reason=f"integration data not found in {DATA_DIR}/",
)

def compute_metrics(truth_file: Path, eris_loci: list, families: dict[str, str] | None = None, count: str = 'max'):
    """
    Evaluates the IS element copy counts against the ground truth. See `count_copies` for the arguments.
    """
    gt_counts, eris_counts = count_copies(truth_file, eris_loci, families, count)

    # Calculate TP, FP, FN
    all_is = set(gt_counts.keys()).union(set(eris_counts.keys()))
    tp = fp = fn = 0

    for is_name in sorted(all_is):
        gt_c = gt_counts.get(is_name, 0)
        er_c = eris_counts.get(is_name, 0)

        # Copy-number-aware metric logic
        c_tp = min(gt_c, er_c)
        c_fp = max(0, er_c - gt_c)
        c_fn = max(0, gt_c - er_c)
        tp += c_tp
        fp += c_fp
        fn += c_fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def count_copies(truth_file: Path, eris_loci: list, families: dict[str, str] | None = None,
                 count: str = 'max') -> tuple[dict[str, int], dict[str, int]]:
    """
    Returns the (truth, eris) IS copy counts, keyed by IS name.

    If `families` maps IS names to families, counts are compared per family, so cross-mapping
    relatives (e.g. IS600 called where the truth has ISSd1, both IS3) are not scored as errors.

    `count` sets how an IS's loci become a copy count: 'max' takes max(#loci, max copy number), which
    forgives a locus reported twice; 'sum' adds up every locus's copy number, as a user summing the report would.
    """
    # Parse Ground Truth
    gt_counts = {}
    with open(truth_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                # Extract base name: 'ISEc23_1' -> 'ISEc23'
                is_name = parts[3].split('_')[0]
                gt_counts[is_name] = gt_counts.get(is_name, 0) + 1

    # Parse Eris Output
    eris_loci_dict = {}
    eris_depth = {}
    eris_sum = {}
    
    for locus in eris_loci:
        if not locus.targets: continue
        
        # Extract base name from the locus target feature ID
        # e.g., 'IS30_1_stitched' -> 'IS30'
        first_target = locus.targets[0].id.split(',')[0]
        is_name = first_target.split('_')[0]
        
        if is_name not in eris_loci_dict:
            eris_loci_dict[is_name] = set()
        eris_loci_dict[is_name].add(locus.id)
        
        # Track the max copy number depth estimate seen for this IS
        eris_depth[is_name] = max(eris_depth.get(is_name, 0), int(locus.fractional_depth))
        eris_sum[is_name] = eris_sum.get(is_name, 0) + int(locus.fractional_depth)

    # Resolve Final Copy Counts for Eris
    eris_counts = {}
    for is_name in eris_loci_dict:
        # Eris copy count = max of the depth estimate OR the distinct physical loci anchored
        eris_counts[is_name] = max(len(eris_loci_dict[is_name]), eris_depth[is_name]) if count == 'max' else eris_sum[is_name]

    # Sum the per-name counts within each family, after the per-name copy estimate above
    if families:
        gt_counts, eris_counts = _sum_by_family(gt_counts, families), _sum_by_family(eris_counts, families)

    return gt_counts, eris_counts


def find_overlapping_claims(eris_loci: list, families: dict[str, str]) -> list[tuple[str, str, str]]:
    """
    Returns (locus_a, locus_b, contig) for every two loci of the same IS family whose targets claim overlapping
    sequence on a contig. Each region of the genome holds at most one copy of an IS, so one should never be reported twice.
    """
    claims = {}  # (family, contig) -> [(start, end, locus_id)]
    for locus in eris_loci:
        if not locus.targets: continue
        name = locus.targets[0].id.split('_')[0]
        for seg in locus.targets[0].segments:
            claims.setdefault((families.get(name, name), seg.contig), []).append((seg.start, seg.end, locus.id))

    overlaps = set()
    for (_, contig), spans in claims.items():
        spans.sort()
        for i, (_, end_a, id_a) in enumerate(spans):
            for start_b, _, id_b in spans[i + 1:]:
                if start_b >= end_a: break
                if id_a != id_b:
                    overlaps.add((*sorted((id_a, id_b)), contig))
    return sorted(overlaps)


def _sum_by_family(counts: dict[str, int], families: dict[str, str]) -> dict[str, int]:
    summed = {}
    for name, n in counts.items():
        family = families.get(name, name)
        summed[family] = summed.get(family, 0) + n
    return summed


@pytest.fixture(scope="module")
def target_db():
    # Only initialize the Minimap2 database once for the entire module to save time
    return TargetDatabase(DATA_DIR / "IS.mmi", FeatureType.CDS, indexing_threads=1)


@pytest.fixture(scope="module")
def families(target_db):
    # Target names are '<name>_<family>_<group>', e.g. 'IS600_IS3_IS3'
    return {n.rsplit('_', 2)[0]: n.rsplit('_', 2)[1] for n in target_db.aligner.seq_names if n.count('_') >= 2}


# User mentioned EC958.fasta but that file only contains the chromosome.
# The actual complete genome (chromosome + plasmids) is EC958_complete_genome.fasta.
BENCHMARKS = {
    "complete": ("EC958_complete_genome.fasta", "complete_ec958_IS.reanotated.sorted.bed"),
    # Fragmented graphs are incredibly messy, but eris should still achieve >85% recovery
    "fragmented": ("contigs_shovill_simulated.gfa", "ec958_IS.reannotated.sorted.bed"),
}

# Families whose summed copy count is still wrong, as {family: (truth, eris)}. Every other family must match the
# truth exactly. A family may get closer to its truth but never further away; update this table when one improves.
KNOWN_COUNT_ERRORS = {
    "complete": {"IS3": (14, 10), "IS5": (1, 0), "IS66": (8, 10)},
    "fragmented": {"IS1": (5, 6), "IS3": (14, 12), "IS5": (1, 0), "IS66": (7, 10)},
}


@pytest.fixture(scope="module", params=BENCHMARKS)
def benchmark(request, target_db):
    genome_file, truth_file = BENCHMARKS[request.param]
    # max_feature_hops=0 completely disables the PyFGS gene calling for extreme speed
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        loci = list(pipeline(GenomeAssembly.from_file(DATA_DIR / genome_file)))
    return request.param, DATA_DIR / truth_file, loci


@needs_data
def test_benchmark_precision_recall(benchmark, families):
    _, truth_file, loci = benchmark

    # IS variants (e.g., IS1A vs IS1R) cross-map heavily in nucleotide space,
    # so exact name matches are impossible; score per IS family and report exact names for reference.
    print("exact-name P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, loci))
    print("family max P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, loci, families))
    precision, recall, f1 = compute_metrics(truth_file, loci, families, count='sum')
    print("family sum P=%.3f R=%.3f F1=%.3f" % (precision, recall, f1))

    # Score summed copy numbers: the max-based count cannot see an IS reported twice
    assert f1 >= 0.85, f"Family F1 on summed copy numbers must be >= 0.85, got {f1:.2f}"


@needs_data
def test_benchmark_family_copy_counts_do_not_regress(benchmark, families):
    # The F1 threshold hides a single double-counted IS (issue #1 scored 0.857); compare each family instead
    name, truth_file, loci = benchmark
    gt_counts, eris_counts = count_copies(truth_file, loci, families, count='sum')
    known = KNOWN_COUNT_ERRORS[name]

    worse = {}
    for family in sorted(set(gt_counts) | set(eris_counts)):
        truth, eris = gt_counts.get(family, 0), eris_counts.get(family, 0)
        allowed = abs(known[family][1] - known[family][0]) if family in known else 0
        if abs(eris - truth) > allowed:
            worse[family] = (truth, eris)
    assert worse == {}, f"(truth, eris) copy counts got worse; known errors are {known}"


@needs_data
def test_benchmark_reports_each_copy_once(benchmark, families):
    _, _, loci = benchmark
    assert find_overlapping_claims(loci, families) == []


def test_compute_metrics_family_level_merges_cross_mapping_relatives(tmp_path):
    # Truth has one IS600 and one ISSd1 (both IS3 family); eris calls the shared node IS600 with 2 copies
    truth = tmp_path / "truth.bed"
    truth.write_text("chr\t0\t1266\tISSd1\t0\t+\nchr\t5000\t6265\tIS600\t0\t-\n")
    stitched = SimpleNamespace(id="locus_1", fractional_depth=2.0,
                               targets=[SimpleNamespace(id="IS600_IS3_IS3_stitched")])
    families = {"IS600": "IS3", "ISSd1": "IS3"}

    assert compute_metrics(truth, [stitched]) == (0.5, 0.5, 0.5)
    assert compute_metrics(truth, [stitched], families=families) == (1.0, 1.0, 1.0)


def test_find_overlapping_claims_flags_leftover_half_of_stitched_locus():
    # Issue #1: the first half of ISEc23 on ctgA was reported both inside the stitch and as a local locus
    def locus(id_, *segments):
        return SimpleNamespace(id=id_, targets=[SimpleNamespace(
            id="ISEc23_IS66_unknown", segments=[SimpleNamespace(contig=c, start=a, end=b) for c, a, b in segments])])

    stitched = locus("locus_stitched", ("ctgA", 0, 1326), ("ctgB", 0, 1135))
    leftover = locus("locus_local", ("ctgA", 0, 1326))
    elsewhere = locus("locus_other", ("ctgA", 2000, 3000))
    families = {"ISEc23": "IS66"}

    assert find_overlapping_claims([stitched, leftover, elsewhere], families) == [("locus_local", "locus_stitched", "ctgA")]
    assert find_overlapping_claims([stitched, elsewhere], families) == []
