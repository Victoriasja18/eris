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

def compute_metrics(truth_file: Path, eris_loci: list, families: dict[str, str] | None = None):
    """
    Evaluates the IS element copy counts against the ground truth.

    If `families` maps IS names to families, counts are compared per family, so cross-mapping
    relatives (e.g. IS600 called where the truth has ISSd1, both IS3) are not scored as errors.
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

    # Resolve Final Copy Counts for Eris
    eris_counts = {}
    for is_name in eris_loci_dict:
        # Eris copy count = max of the depth estimate OR the distinct physical loci anchored
        eris_counts[is_name] = max(len(eris_loci_dict[is_name]), eris_depth[is_name])

    # Sum the per-name counts within each family, after the per-name copy estimate above
    if families:
        gt_counts, eris_counts = _sum_by_family(gt_counts, families), _sum_by_family(eris_counts, families)

    # Calculate TP, FP, FN
    all_is = set(gt_counts.keys()).union(set(eris_counts.keys()))
    tp = fp = fn = 0

    for is_name in sorted(all_is):
        gt_c = gt_counts.get(is_name, 0)
        er_c = eris_counts.get(is_name, 0)

        # Abundance-aware metric logic
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


@needs_data
def test_complete_genome_precision_recall(target_db, families):
    # User mentioned EC958.fasta but that file only contains the chromosome.
    # The actual complete genome (chromosome + plasmids) is EC958_complete_genome.fasta.
    genome = GenomeAssembly.from_file(DATA_DIR / "EC958_complete_genome.fasta")
    truth_file = DATA_DIR / "complete_ec958_IS.reanotated.sorted.bed"
    
    # max_feature_hops=0 completely disables the PyFGS gene calling for extreme speed
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        loci = list(pipeline(genome))
        
    # IS variants (e.g., IS1A vs IS1R) cross-map heavily in nucleotide space,
    # so exact name matches are impossible; score per IS family and report exact names for reference.
    print("exact-name P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, loci))
    precision, recall, f1 = compute_metrics(truth_file, loci, families)
    print("family     P=%.3f R=%.3f F1=%.3f" % (precision, recall, f1))

    assert f1 >= 0.85, f"F1 score on complete genome must be >= 0.85, got {f1:.2f}"


@needs_data
def test_fragmented_assembly_precision_recall(target_db, families):
    genome = GenomeAssembly.from_file(DATA_DIR / "contigs_shovill_simulated.gfa")
    truth_file = DATA_DIR / "ec958_IS.reannotated.sorted.bed"
    
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        loci = list(pipeline(genome))
        
    print("exact-name P=%.3f R=%.3f F1=%.3f" % compute_metrics(truth_file, loci))
    precision, recall, f1 = compute_metrics(truth_file, loci, families)
    print("family     P=%.3f R=%.3f F1=%.3f" % (precision, recall, f1))

    # Fragmented graphs are incredibly messy, but eris should still achieve >85% recovery
    assert f1 >= 0.85, f"F1 score on fragmented assembly must be >= 0.85, got {f1:.2f}"


def test_compute_metrics_family_level_merges_cross_mapping_relatives(tmp_path):
    # Truth has one IS600 and one ISSd1 (both IS3 family); eris calls the shared node IS600 with 2 copies
    truth = tmp_path / "truth.bed"
    truth.write_text("chr\t0\t1266\tISSd1\t0\t+\nchr\t5000\t6265\tIS600\t0\t-\n")
    stitched = SimpleNamespace(id="locus_1", fractional_depth=2.0,
                               targets=[SimpleNamespace(id="IS600_IS3_IS3_stitched")])
    families = {"IS600": "IS3", "ISSd1": "IS3"}

    assert compute_metrics(truth, [stitched]) == (0.5, 0.5, 0.5)
    assert compute_metrics(truth, [stitched], families=families) == (1.0, 1.0, 1.0)
