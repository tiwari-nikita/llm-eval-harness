from taxonomy import classify, word_count, length_band


def test_code_bucket_wins_over_generic_question():
    assert classify("Why is my python script throwing a traceback?") == "code"


def test_advisory_pattern():
    assert classify("Should I take the higher paying offer or not?") == "advisory"


def test_personal_pattern():
    assert classify("My relationship with my mum has been rough lately") == "personal"


def test_roleplay_pattern():
    assert classify("Pretend you are a pirate and speak like one") == "roleplay"


def test_falls_back_to_factual_for_bare_question():
    assert classify("What's the capital of Peru?") == "factual"


def test_falls_back_to_other():
    assert classify("thanks, that helps") == "other"


def test_empty_text_is_other():
    assert classify("") == "other"
    assert classify(None) == "other"


def test_order_matters_code_beats_advisory():
    # contains both "should i" (advisory) and "debug" (code) -- code rules
    # come first in taxonomy.RULES, so code should win.
    assert classify("should i debug this function myself or hire someone") == "code"


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0
    assert word_count(None) == 0


def test_length_band_boundaries():
    assert length_band(9) == "very_short"
    assert length_band(10) == "short"
    assert length_band(39) == "short"
    assert length_band(40) == "medium"
    assert length_band(149) == "medium"
    assert length_band(150) == "long"
