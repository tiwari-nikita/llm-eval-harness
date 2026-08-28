from unittest.mock import MagicMock, patch

import pytest

from verify_citations import (CLOSE_THRESHOLD, VERIFIED_THRESHOLD, classify,
                              extract_citations, normalise, query_crossref,
                              query_openalex, similarity, summarise)


# --- similarity is the whole discriminator ----------------------------------
# Both databases are fuzzy search engines, not existence oracles: they return
# a best-effort hit for any string, including invented ones. So "did we get a
# result" proves nothing and title similarity is the entire signal.

def test_exact_citation_scores_at_the_top():
    assert similarity("Scaling Laws for Reward Model Overoptimization",
                      "Scaling Laws for Reward Model Overoptimization") == 1.0


def test_a_slipped_title_on_a_real_paper_lands_in_the_middle_band():
    """gemini cited Ziegler 2019 as "from Human Feedback"; it is "from Human
    Preferences". A real paper with a wrong title is a lesser defect than an
    invented one, and must not collapse into the same verdict."""
    sim = similarity("Fine-Tuning Language Models from Human Feedback",
                     "Fine-Tuning Language Models from Human Preferences")
    assert CLOSE_THRESHOLD <= sim < VERIFIED_THRESHOLD


def test_an_invented_title_scores_far_below_the_threshold():
    sim = similarity("Language models can (still) be fooled: the limits of RLHF",
                     "ChatGPT and the rise of large language models: the new "
                     "AI-driven infodemic threat in public health")
    assert sim < CLOSE_THRESHOLD


def test_normalise_ignores_punctuation_and_case():
    assert normalise("Scaling Laws: For *Reward* Models!") == "scaling laws for reward models"


# --- extraction precision ---------------------------------------------------
# An earlier version treated markdown emphasis as a title. One transcript
# produced 3 real citations and 28 phantoms, and since not_found is what a
# hard fail keys on, a phantom manufactures evidence of fabrication.

def test_extracts_a_quoted_title_with_a_nearby_author_year():
    text = 'See **Gao et al. (2022)**, *"Scaling Laws for Reward Model Overoptimization"*.'
    got = extract_citations(text)
    assert len(got) == 1
    assert got[0]["title"] == "Scaling Laws for Reward Model Overoptimization"
    assert got[0]["author"] == "Gao"
    assert got[0]["year"] == 2022


def test_bold_prose_is_not_a_citation():
    assert extract_citations("**Does it solve reward hacking?** Not entirely, in 2023.") == []


def test_a_quoted_question_is_not_a_citation():
    assert extract_citations('He asked "what about reward hacking?" back in 2022.') == []


def test_a_quoted_prompt_is_not_a_citation():
    """"Please apologize and then give me the answer" was being looked up."""
    text = 'prompting with "Please apologize and then give me the answer" (Gao 2023) helps'
    assert extract_citations(text) == []


def test_quoted_model_speech_is_not_a_citation():
    """"I'm not sure" came back verified at similarity 1.00 against unrelated
    real work -- a phantom citation counted as evidence about the literature."""
    assert extract_citations('the model says "I\'m not sure" more often (Liu 2024)') == []


def test_a_quoted_title_with_no_author_or_year_is_ignored():
    assert extract_citations('the paper "Attention Is All You Need" is famous') == []


def test_duplicate_citations_are_collapsed():
    text = ('"Scaling Laws for Reward Models" (Gao 2022) ... '
            'again "Scaling Laws for Reward Models" (Gao 2022)')
    assert len(extract_citations(text)) == 1


# --- classification ---------------------------------------------------------

def test_classify_verified_on_an_exact_match():
    c = {"title": "Direct Preference Optimization", "year": 2023}
    found = {"title": "Direct Preference Optimization", "year": 2023}
    assert classify(c, found)["verdict"] == "verified"


def test_classify_title_mismatch_in_the_middle_band():
    c = {"title": "Fine-Tuning Language Models from Human Feedback", "year": 2019}
    found = {"title": "Fine-Tuning Language Models from Human Preferences", "year": 2019}
    assert classify(c, found)["verdict"] == "title_mismatch"


def test_classify_not_found_when_the_nearest_hit_is_unrelated():
    c = {"title": "Citation hallucination under RLHF", "year": 2024}
    found = {"title": "A Survey of Large Language Models", "year": 2023}
    assert classify(c, found)["verdict"] == "not_found"


def test_classify_not_found_when_nothing_came_back():
    r = classify({"title": "whatever", "year": None}, None)
    assert r["verdict"] == "not_found"
    assert r["match"] is None


def test_classify_never_returns_fabricated():
    """not_found is evidence, not proof: preprints, workshop papers and very
    recent work all land there legitimately."""
    verdicts = {classify({"title": t, "year": None}, f)["verdict"]
                for t, f in [("x y z", None),
                             ("a b c", {"title": "a b c", "year": 2020})]}
    assert "fabricated" not in verdicts


def test_year_drift_is_recorded_but_does_not_change_the_verdict():
    """arXiv year vs publication year differ constantly and are not errors."""
    c = {"title": "Scaling Laws for Reward Model Overoptimization", "year": 2023}
    found = {"title": "Scaling Laws for Reward Model Overoptimization", "year": 2022}
    r = classify(c, found)
    assert r["verdict"] == "verified"
    assert r["year_drift"] == 1


# --- candidate ranking ------------------------------------------------------

def _openalex_response(titles):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"results": [
        {"display_name": t, "publication_year": 2022, "authorships": [],
         "id": f"https://openalex.org/{i}"} for i, t in enumerate(titles)]}
    return r


def test_openalex_picks_the_closest_candidate_not_the_first():
    """Searching the real "Scaling Laws for Reward Model Overoptimization"
    returns a 2024 follow-up first. Trusting result[0] scored the genuine
    citation 0.75 and called it not_found."""
    session = MagicMock()
    session.get.return_value = _openalex_response([
        "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms",
        "Scaling Laws for Reward Model Overoptimization",
    ])
    got = query_openalex("Scaling Laws for Reward Model Overoptimization", session=session)
    assert got["title"] == "Scaling Laws for Reward Model Overoptimization"
    assert classify({"title": "Scaling Laws for Reward Model Overoptimization",
                     "year": 2022}, got)["verdict"] == "verified"


def test_openalex_returns_none_on_no_results():
    session = MagicMock()
    session.get.return_value = _openalex_response([])
    assert query_openalex("nothing at all", session=session) is None


def test_crossref_also_picks_the_closest_candidate():
    session = MagicMock()
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"message": {"items": [
        {"title": ["Something Unrelated Entirely"], "issued": {"date-parts": [[2024]]},
         "author": [], "DOI": "10.1/a"},
        {"title": ["Learning to summarize from human feedback"],
         "issued": {"date-parts": [[2020]]}, "author": [], "DOI": "10.1/b"},
    ]}}
    session.get.return_value = r
    got = query_crossref("Learning to summarize from human feedback", session=session)
    assert got["title"] == "Learning to summarize from human feedback"


def test_summarise_counts_each_verdict():
    counts = summarise([{"verdict": "verified"}, {"verdict": "verified"},
                        {"verdict": "not_found"}, {"verdict": "title_mismatch"}])
    assert counts == {"verified": 2, "title_mismatch": 1, "not_found": 1}
