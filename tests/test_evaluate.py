"""The metrics must be right, because every conclusion rests on them."""

from causalpdm.pipeline import evaluate


def test_a_perfect_graph_scores_perfectly():
    edges = {("a", "b"), ("b", "c")}
    score = evaluate.score_graph(edges, edges)
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f1"] == 1.0
    assert score["shd"] == 0


def test_an_empty_graph_scores_zero():
    score = evaluate.score_graph(set(), {("a", "b")})
    assert score["recall"] == 0.0
    assert score["f1"] == 0.0
    assert score["missed"] == 1


def test_counting_of_right_wrong_and_missed():
    found = {("a", "b"), ("x", "y")}
    expert = {("a", "b"), ("b", "c")}
    score = evaluate.score_graph(found, expert)
    assert score["correct"] == 1      # a->b
    assert score["wrong"] == 1        # x->y was invented
    assert score["missed"] == 1       # b->c was not found
    assert score["shd"] == 2


def test_wrong_direction_counts_as_wrong_but_not_as_missing_the_link():
    """
    Pointing a real link backwards is a different mistake from not finding it,
    and the two scores together tell us which one happened.
    """
    found = {("b", "a")}
    expert = {("a", "b")}
    assert evaluate.score_graph(found, expert)["correct"] == 0
    assert evaluate.score_graph_ignoring_direction(found, expert)["correct"] == 1


def test_precision_at_k_counts_only_the_top_guesses():
    assert evaluate.precision_at_k(["a", "b", "c"], ["a"], k=3) == 1 / 3
    assert evaluate.precision_at_k(["a", "b", "c"], ["a", "b"], k=2) == 1.0
    assert evaluate.precision_at_k(["x", "y", "z"], ["a"], k=3) == 0.0


def test_recall_at_k_counts_how_many_causes_we_caught():
    assert evaluate.recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0
    assert evaluate.recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5


def test_average_precision_rewards_putting_the_answer_first():
    first = evaluate.average_precision_at_k(["a", "x", "y"], ["a"], k=3)
    third = evaluate.average_precision_at_k(["x", "y", "a"], ["a"], k=3)
    assert first > third
    assert first == 1.0


def test_reciprocal_rank_is_one_over_the_position():
    assert evaluate.reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert evaluate.reciprocal_rank(["b", "a"], ["a"]) == 0.5
    assert evaluate.reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_scoring_many_faults_averages_them():
    results = [(["a", "b", "c"], ["a"]), (["x", "y", "z"], ["a"])]
    scores = evaluate.score_rankings(results, k=3)
    assert scores["faults_scored"] == 2
    assert scores["found_in_top_1"] == 0.5   # right on the first, wrong on the second
