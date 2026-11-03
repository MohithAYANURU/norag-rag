from v2_app import app, split_text, word_analysis


def test_split_text_respects_overlap():
    chunks = split_text("a" * 2500, 800, 10)
    assert len(chunks) == 4
    assert all(0 < len(chunk) <= 800 for chunk in chunks)


def test_word_analysis_returns_question_words():
    result = word_analysis(["own:vellora-duo-espresso-machine.txt"], None, "descale vinegar")
    values = {item["word"]: item for item in result["words"]}
    assert values["descale"]["exists"]
    assert values["vinegar"]["exists"]


def test_about_route_is_not_registered():
    assert all(route.path != "/about" for route in app.routes)
