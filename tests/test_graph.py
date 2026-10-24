import pytest
import numpy as np

from eris.graph import Graph, Edge, TopologyEngine
from eris.constants import Strand
from eris.alignment import AlignmentBatch

def test_undirected_graph_edges():
    graph = Graph(directed=False)
    # Add an edge from ctgA to ctgB
    edge = Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=50)
    graph.add_edge(edge)
    
    # We should be able to get neighbors for ctgA (the direct edge)
    a_neighbors = graph.get_neighbors("ctgA")
    assert len(a_neighbors) == 1
    assert list(a_neighbors)[0].v == "ctgB"
    
    # And we should be able to get neighbors for ctgB (the implicit reverse edge)
    b_neighbors = graph.get_neighbors("ctgB")
    assert len(b_neighbors) == 1
    reverse_edge = list(b_neighbors)[0]
    
    assert reverse_edge.v == "ctgA"
    assert reverse_edge.u_strand == Strand.FORWARD
    assert reverse_edge.overlap == 50

def test_topology_engine_bounded_paths():
    # Construct a mock DAG: ctgA -> ctgB -> ctgC
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0),
        Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0)
    ]
    contig_lengths = {"ctgA": 1000, "ctgB": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB": 2.0, "ctgC": 10.0}
    
    engine = TopologyEngine(edges, contig_lengths, contig_depths)
    
    # Expected length across B is exactly 500bp
    paths = engine._find_bounded_paths(
        start_ctg="ctgA", start_strand=Strand.FORWARD,
        target_ctg="ctgC", target_strand=Strand.FORWARD,
        expected_len=500, tolerance=100
    )
    
    assert len(paths) == 1
    p = paths[0]
    assert p['contigs'] == ["ctgA", "ctgB", "ctgC"]
    assert p['length'] == 500  # The path walked across B entirely
    assert p['min_depth'] == 2.0  # Bottleneck depth was on ctgB

def test_topology_engine_collapsed_path():
    # Branching path ctgA -> ctgB1 (depth=1) -> ctgC
    #                ctgA -> ctgB2 (depth=10) -> ctgC
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB1", Strand.FORWARD, overlap=0),
        Edge("ctgB1", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0),
        Edge("ctgA", Strand.FORWARD, "ctgB2", Strand.FORWARD, overlap=0),
        Edge("ctgB2", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0),
    ]
    contig_lengths = {"ctgA": 1000, "ctgB1": 500, "ctgB2": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB1": 1.0, "ctgB2": 10.0, "ctgC": 10.0}
    
    engine = TopologyEngine(edges, contig_lengths, contig_depths)
    
    # The greedy path finder should pick B2 because it has the highest depth (10.0 > 1.0)
    paths = engine._find_collapsed_path(
        start_ctg="ctgA", start_strand=Strand.FORWARD,
        target_ctg="ctgC", target_strand=Strand.FORWARD,
        expected_len=500, tolerance=100
    )
    
    assert len(paths) == 1
    assert paths[0]['contigs'] == ["ctgA", "ctgB2", "ctgC"]
    assert paths[0]['min_depth'] == 10.0

def test_topology_engine_resolve():
    # Simulate a shattered alignment:
    # An IS element maps perfectly to the end of ctgA, and the beginning of ctgC
    edges = [
        Edge("ctgA", Strand.FORWARD, "ctgB", Strand.FORWARD, overlap=0),
        Edge("ctgB", Strand.FORWARD, "ctgC", Strand.FORWARD, overlap=0)
    ]
    contig_lengths = {"ctgA": 1000, "ctgB": 500, "ctgC": 1000}
    contig_depths = {"ctgA": 10.0, "ctgB": 10.0, "ctgC": 10.0}
    
    engine = TopologyEngine(edges, contig_lengths, contig_depths, mode='variant')
    
    # Create two partial AlignmentRecords representing the shattered hit
    batch_a = AlignmentBatch(
        q_names=np.array(["IS1"]), q_lengths=np.array([2000], dtype=np.int32),
        q_starts=np.array([0], dtype=np.int32), q_ends=np.array([1000], dtype=np.int32),
        t_names=np.array(["ctgA"]), t_lengths=np.array([1000], dtype=np.int32),
        t_starts=np.array([0], dtype=np.int32), t_ends=np.array([1000], dtype=np.int32),
        strands=np.array([Strand.FORWARD], dtype=np.int8),
        lengths=np.array([1000], dtype=np.int32), matches=np.array([1000], dtype=np.int32),
        mismatches=np.array([0], dtype=np.int32), qualities=np.array([60], dtype=np.int32),
        cigars=np.array(["1000M"], dtype=object)
    )
    
    batch_c = AlignmentBatch(
        q_names=np.array(["IS1"]), q_lengths=np.array([2000], dtype=np.int32),
        q_starts=np.array([1500], dtype=np.int32), q_ends=np.array([2000], dtype=np.int32),
        t_names=np.array(["ctgC"]), t_lengths=np.array([1000], dtype=np.int32),
        t_starts=np.array([0], dtype=np.int32), t_ends=np.array([500], dtype=np.int32),
        strands=np.array([Strand.FORWARD], dtype=np.int8),
        lengths=np.array([500], dtype=np.int32), matches=np.array([500], dtype=np.int32),
        mismatches=np.array([0], dtype=np.int32), qualities=np.array([60], dtype=np.int32),
        cigars=np.array(["500M"], dtype=object)
    )
    
    alignments = {"ctgA": batch_a, "ctgC": batch_c}
    
    cleaned, paths = engine.resolve_split_alignments(alignments)
    
    # We should successfully stitch 1 multi-contig path over the gap
    assert len(paths) == 1
    stitched_path = paths[0]
    
    # The payload sequence should contain: [real_hit_A, synthetic_hit_B, real_hit_C]
    assert len(stitched_path) == 3
    assert stitched_path[0].t_name == "ctgA"
    assert stitched_path[1].t_name == "ctgB"
    assert stitched_path[1].idx == -1 # Verify it's a mock record for the unaligned node
    assert stitched_path[2].t_name == "ctgC"
    
    # Because both hits were stitched into the path, the 'cleaned' pool should be empty
    assert len(cleaned) == 0
