from router import category_of, rank_objective, rank_pairwise


def test_category_of():
    assert category_of("advisory_001") == "advisory"
    assert category_of("research_002") == "research"
    assert category_of("personal_003") == "personal"


def test_rank_objective_averages_and_sorts_descending():
    scores = [
        {"task_id": "research_001", "provider": "groq", "model": "a", "score": 1.0},
        {"task_id": "research_002", "provider": "groq", "model": "a", "score": 0.5},
        {"task_id": "research_001", "provider": "groq", "model": "b", "score": 0.2},
        {"task_id": "documents_001", "provider": "groq", "model": "a", "score": 0.0},  # different category
    ]
    rows = rank_objective("research", scores)
    assert rows[0]["provider"] == "groq" and rows[0]["model"] == "a"
    assert rows[0]["mean_score"] == 0.75
    assert rows[0]["n_tasks"] == 2
    assert rows[1]["model"] == "b"
    assert rows[1]["n_tasks"] == 1


def test_rank_objective_empty():
    assert rank_objective("research", []) == []


def test_rank_pairwise_counts_wins_losses_ties():
    votes = [
        {"task_id": "advisory_001", "model_a": "m1", "model_b": "m2", "votes": {"overall": "a"}},
        {"task_id": "advisory_002", "model_a": "m2", "model_b": "m1", "votes": {"overall": "a"}},
        {"task_id": "advisory_003", "model_a": "m1", "model_b": "m2", "votes": {"overall": "tie"}},
        {"task_id": "research_001", "model_a": "m1", "model_b": "m2", "votes": {"overall": "a"}},  # wrong category, excluded
    ]
    rows = {r["model"]: r for r in rank_pairwise("advisory", votes)}
    assert rows["m1"]["wins"] == 1
    assert rows["m1"]["losses"] == 1
    assert rows["m1"]["ties"] == 1
    assert rows["m1"]["win_rate"] == 0.5
    assert rows["m2"]["wins"] == 1
    assert rows["m2"]["losses"] == 1


def test_rank_pairwise_no_decisive_games_gives_none_win_rate():
    votes = [{"task_id": "advisory_001", "model_a": "m1", "model_b": "m2", "votes": {"overall": "tie"}}]
    rows = {r["model"]: r for r in rank_pairwise("advisory", votes)}
    assert rows["m1"]["win_rate"] is None
