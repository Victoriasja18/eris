import pytest
import numpy as np

from eris.alignment import Cigar, AlignmentBatch
from eris.constants import Strand

def test_cigar_parsing():
    cigar = Cigar("10M2I5M")
    ops = list(cigar)
    
    # Operations should be evaluated chronologically:
    # 1. 10M: q_len=10, t_len=10, aln_len=10
    # 2. 2I:  q_len=12, t_len=10, aln_len=10
    # 3. 5M:  q_len=17, t_len=15, aln_len=15
    assert len(ops) == 3
    assert ops[0] == ('M', 10, 10, 10, 10)
    assert ops[1] == ('I', 2, 12, 10, 10)
    assert ops[2] == ('M', 5, 17, 15, 15)

def _mock_alignment_batch():
    # Helper to generate a mock AlignmentBatch for downstream tests
    return AlignmentBatch(
        q_names=np.array(["read1", "read1", "read2"]),
        q_lengths=np.array([100, 100, 150], dtype=np.int32),
        q_starts=np.array([0, 50, 10], dtype=np.int32),
        q_ends=np.array([40, 90, 60], dtype=np.int32),
        t_names=np.array(["ctgA", "ctgB", "ctgA"]),
        t_lengths=np.array([1000, 2000, 1000], dtype=np.int32),
        t_starts=np.array([100, 500, 110], dtype=np.int32),
        t_ends=np.array([140, 540, 160], dtype=np.int32),
        strands=np.array([Strand.FORWARD, Strand.REVERSE, Strand.FORWARD], dtype=np.int8),
        lengths=np.array([40, 40, 50], dtype=np.int32),
        matches=np.array([40, 38, 48], dtype=np.int32),
        mismatches=np.array([0, 2, 2], dtype=np.int32),
        qualities=np.array([60, 30, 40], dtype=np.int32),
        cigars=np.array(["40M", "38M2X", "48M2X"], dtype=object)
    )

def test_alignment_batch_swap_sides():
    batch = _mock_alignment_batch()
    swapped = batch.swap_sides()
    
    # Query arrays should physically swap with Target arrays
    assert np.array_equal(swapped.q_names, batch.t_names)
    assert np.array_equal(swapped.q_lengths, batch.t_lengths)
    assert np.array_equal(swapped.q_starts, batch.t_starts)
    
    assert np.array_equal(swapped.t_names, batch.q_names)
    assert np.array_equal(swapped.t_starts, batch.q_starts)
    
    # Check CIGAR translation logic for reciprocal mapping
    batch_with_indels = AlignmentBatch(
        q_names=np.array(["read1"]), q_lengths=np.array([100], dtype=np.int32),
        q_starts=np.array([0], dtype=np.int32), q_ends=np.array([10], dtype=np.int32),
        t_names=np.array(["ctgA"]), t_lengths=np.array([1000], dtype=np.int32),
        t_starts=np.array([100], dtype=np.int32), t_ends=np.array([110], dtype=np.int32),
        strands=np.array([Strand.FORWARD], dtype=np.int8),
        lengths=np.array([10], dtype=np.int32), matches=np.array([10], dtype=np.int32),
        mismatches=np.array([0], dtype=np.int32), qualities=np.array([60], dtype=np.int32),
        cigars=np.array(["5M2I3M1D"], dtype=object)
    )
    
    swapped_indels = batch_with_indels.swap_sides()
    # Insertions relative to target become deletions, and vice-versa
    assert swapped_indels.cigars[0] == "5M2D3M1I"

def test_alignment_batch_cull_overlaps():
    # Give two competing alignments on the same target that overlap heavily
    # Hit 0: 100-200 (score 100) -> Should win
    # Hit 1: 150-250 (score 80) -> Should be destroyed
    batch = AlignmentBatch(
        q_names=np.array(["r1", "r2"]),
        q_lengths=np.array([100, 100], dtype=np.int32),
        q_starts=np.array([0, 0], dtype=np.int32),
        q_ends=np.array([100, 100], dtype=np.int32),
        t_names=np.array(["ctgA", "ctgA"]),
        t_lengths=np.array([1000, 1000], dtype=np.int32),
        t_starts=np.array([100, 150], dtype=np.int32),
        t_ends=np.array([200, 250], dtype=np.int32),
        strands=np.array([Strand.FORWARD, Strand.FORWARD], dtype=np.int8),
        lengths=np.array([100, 100], dtype=np.int32),
        matches=np.array([100, 80], dtype=np.int32),
        mismatches=np.array([0, 0], dtype=np.int32),
        qualities=np.array([60, 60], dtype=np.int32),
        cigars=np.array(["100M", "100M"], dtype=object)
    )
    
    # Since they overlap by 50bp and 50/100 > 0.1 max allowed overlap fraction
    culled = batch.cull_overlaps(max_overlap_fraction=0.1)
    
    assert len(culled) == 1
    assert culled.t_starts[0] == 100
    assert culled.t_ends[0] == 200

def test_alignment_batch_to_intervals():
    batch = _mock_alignment_batch()
    
    # Convert using target coordinates
    intervals = batch.to_intervals(by_query=False) 
    assert len(intervals) == 3
    assert np.array_equal(intervals.starts, batch.t_starts)
    assert np.array_equal(intervals.ends, batch.t_ends)
    assert np.array_equal(intervals.original_indices, np.array([0, 1, 2]))
    
    # Convert using query coordinates
    query_intervals = batch.to_intervals(by_query=True) 
    assert np.array_equal(query_intervals.starts, batch.q_starts)
