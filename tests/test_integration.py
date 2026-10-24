import pytest
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
pytestmark = pytest.mark.skipif(
    not all((DATA_DIR / f).exists() for f in REQUIRED_FILES),
    reason=f"integration data not found in {DATA_DIR}/",
)

def compute_metrics(truth_file: Path, eris_loci: list):
    """Evaluates the IS element copy counts against the ground truth."""
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


@pytest.fixture(scope="module")
def target_db():
    # Only initialize the Minimap2 database once for the entire module to save time
    return TargetDatabase(DATA_DIR / "IS.mmi", FeatureType.CDS, indexing_threads=1)


def test_complete_genome_precision_recall(target_db):
    # User mentioned EC958.fasta but that file only contains the chromosome.
    # The actual complete genome (chromosome + plasmids) is EC958_complete_genome.fasta.
    genome = GenomeAssembly.from_file(DATA_DIR / "EC958_complete_genome.fasta")
    truth_file = DATA_DIR / "complete_ec958_IS.reanotated.sorted.bed"
    
    # max_feature_hops=0 completely disables the PyFGS gene calling for extreme speed
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        loci = list(pipeline(genome))
        
    precision, recall, f1 = compute_metrics(truth_file, loci)
    
    # IS variants (e.g., IS1A vs IS1R) cross-map heavily in nucleotide space, 
    # so exact 1.0 matches are impossible without grouping by IS family. 
    # We check for excellent F1 score instead.
    assert f1 >= 0.85, f"F1 score on complete genome must be >= 0.85, got {f1:.2f}"


def test_fragmented_assembly_precision_recall(target_db):
    genome = GenomeAssembly.from_file(DATA_DIR / "contigs_shovill_simulated.gfa")
    truth_file = DATA_DIR / "ec958_IS.reannotated.sorted.bed"
    
    with Pipeline(target_db, max_feature_hops=0, mode='collapse', min_coverage=0.30) as pipeline:
        loci = list(pipeline(genome))
        
    precision, recall, f1 = compute_metrics(truth_file, loci)
    
    # Fragmented graphs are incredibly messy, but eris should still achieve >85% recovery
    assert f1 >= 0.85, f"F1 score on fragmented assembly must be >= 0.85, got {f1:.2f}"
