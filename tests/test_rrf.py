from rag.rrf import RRF_K, reciprocal_rank_fusion


def test_hand_computed_single_list():
    hits = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    fused = reciprocal_rank_fusion([hits])
    scores = {h["id"]: h["rrf_score"] for h in fused}

    assert scores["a"] == 1 / (RRF_K + 1)
    assert scores["b"] == 1 / (RRF_K + 2)
    assert scores["c"] == 1 / (RRF_K + 3)
    assert [h["id"] for h in fused] == ["a", "b", "c"]  # already sorted desc by score


def test_fuses_more_than_two_lists():
    """The whole point of generalizing beyond hybrid.py's 2-list case: N lists fuse too."""
    list_1 = [{"id": "x"}, {"id": "y"}]
    list_2 = [{"id": "y"}, {"id": "z"}]
    list_3 = [{"id": "z"}, {"id": "y"}]

    fused = reciprocal_rank_fusion([list_1, list_2, list_3])
    result_ids = [h["id"] for h in fused]

    # "y" appears in all 3 lists (ranks 2, 1, 2) -> highest combined score.
    assert result_ids[0] == "y"
    assert set(result_ids) == {"x", "y", "z"}


def test_hit_payload_is_preserved_not_just_id():
    hits = [{"id": "a", "text": "hello", "metadata": {"ticker": "GOOGL"}}]
    fused = reciprocal_rank_fusion([hits])
    assert fused[0]["text"] == "hello"
    assert fused[0]["metadata"] == {"ticker": "GOOGL"}
