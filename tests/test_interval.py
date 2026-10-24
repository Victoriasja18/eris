import pytest
import numpy as np

from eris.interval import Interval, IntervalBatch
from eris.constants import Strand, Context

def test_interval_contains():
    iv = Interval(10, 20, Strand.FORWARD)
    
    # Test points
    assert 15 in iv
    assert 10 in iv
    assert 20 not in iv  # exclusive end
    
    # Test fully contained intervals
    iv2 = Interval(12, 18, Strand.FORWARD)
    assert iv2 in iv
    
    # Test overlapping but not fully contained
    iv3 = Interval(5, 15, Strand.FORWARD)
    assert iv3 not in iv

def test_interval_shift():
    iv = Interval(10, 20, Strand.FORWARD)
    shifted = iv.shift(5)
    
    assert shifted.start == 15
    assert shifted.end == 25
    assert shifted.strand == Strand.FORWARD

def test_interval_reverse_complement():
    # Given a sequence of length 100
    iv = Interval(10, 20, Strand.FORWARD)
    rc = iv.reverse_complement(length=100)
    
    assert rc.start == 80
    assert rc.end == 90
    assert rc.strand == Strand.REVERSE

def test_interval_relate():
    target = Interval(50, 100, Strand.FORWARD)
    
    # Gene strictly upstream
    upstream = Interval(10, 30, Strand.FORWARD)
    assert target.relate(upstream) == Context.UPSTREAM
    
    # Gene strictly downstream
    downstream = Interval(120, 150, Strand.FORWARD)
    assert target.relate(downstream) == Context.DOWNSTREAM
    
    # Gene strictly inside
    inside = Interval(60, 80, Strand.FORWARD)
    assert target.relate(inside) == Context.INSIDE

def test_interval_batch_sort():
    starts = [30, 10, 20]
    ends = [40, 15, 25]
    strands = [Strand.FORWARD] * 3
    
    batch = IntervalBatch(
        starts=np.array(starts, dtype=np.int32), 
        ends=np.array(ends, dtype=np.int32), 
        strands=np.array(strands, dtype=np.int8)
    )
    
    sorted_batch = batch.sort()
    
    assert np.array_equal(sorted_batch.starts, np.array([10, 20, 30], dtype=np.int32))
    assert np.array_equal(sorted_batch.ends, np.array([15, 25, 40], dtype=np.int32))
    # Original indices should be tracked!
    assert np.array_equal(sorted_batch.original_indices, np.array([1, 2, 0], dtype=np.int32))

def test_interval_batch_query():
    intervals = [
        Interval(10, 20),
        Interval(30, 40),
        Interval(50, 60)
    ]
    # Sorting is mathematically required before querying!
    batch = IntervalBatch.from_intervals(intervals).sort()
    
    # Query an interval that overlaps the second box (30-40)
    hits = batch.query(35, 45)
    
    assert len(hits) == 1
    assert hits[0] == 1  # Index 1 corresponds to the 30-40 interval

def test_interval_batch_merge():
    intervals = [
        Interval(10, 20),
        Interval(18, 30), # Overlaps with previous
        Interval(40, 50), # Disjoint
        Interval(52, 60), # Close but disjoint
    ]
    batch = IntervalBatch.from_intervals(intervals).sort()
    
    # Merge with 0 tolerance
    merged = batch.merge(tolerance=0)
    assert len(merged) == 3
    assert merged[0].start == 10 and merged[0].end == 30
    assert merged[1].start == 40 and merged[1].end == 50
    assert merged[2].start == 52 and merged[2].end == 60
    
    # Merge with 5 tolerance
    merged_tol = batch.merge(tolerance=5)
    assert len(merged_tol) == 2
    assert merged_tol[0].start == 10 and merged_tol[0].end == 30
    assert merged_tol[1].start == 40 and merged_tol[1].end == 60
