from __future__ import annotations

import pytest

from h3_a100.distributed import resolve_hybrid_topology


def test_two_node_eight_gpu_topology():
    rank0 = resolve_hybrid_topology(world_size=16, rank=0, shard_size=8)
    rank7 = resolve_hybrid_topology(world_size=16, rank=7, shard_size=8)
    rank8 = resolve_hybrid_topology(world_size=16, rank=8, shard_size=8)
    rank15 = resolve_hybrid_topology(world_size=16, rank=15, shard_size=8)

    assert (rank0.replicate_rank, rank0.shard_rank) == (0, 0)
    assert (rank7.replicate_rank, rank7.shard_rank) == (0, 7)
    assert (rank8.replicate_rank, rank8.shard_rank) == (1, 0)
    assert (rank15.replicate_rank, rank15.shard_rank) == (1, 7)
    assert rank15.replicate_size == 2


def test_one_node_smoke_is_one_replica_eight_shards():
    topology = resolve_hybrid_topology(world_size=8, rank=5, shard_size=8)
    assert topology.replicate_size == 1
    assert topology.replicate_rank == 0
    assert topology.shard_rank == 5


@pytest.mark.parametrize(
    "world_size,rank,shard_size",
    [(16, 0, 6), (0, 0, 1), (8, 8, 8), (8, -1, 8)],
)
def test_invalid_topologies_fail(world_size, rank, shard_size):
    with pytest.raises(ValueError):
        resolve_hybrid_topology(world_size, rank, shard_size)
