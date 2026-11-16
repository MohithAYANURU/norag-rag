import pytest

from v2_app import (
    Citation,
    SearchRequest,
    app,
    generate_answer,
    rewrite_follow_up,
    source_glossary,
    suggested_questions,
    split_text,
    word_analysis,
)


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


def test_search_request_rejects_unknown_answer_style():
    with pytest.raises(ValueError):
        SearchRequest(question="What is this?", style="invented")


def test_generate_answer_sends_style_and_history(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": "Answer"})()},
                        )()
                    ]
                },
            )()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(
        "v2_app._llm_client", lambda: (FakeClient(), "test-model")
    )

    answer = generate_answer(
        "What is it?",
        [
            Citation(
                document_id="own:manual.txt",
                source="manual.txt",
                chunk_id="manual.txt:0",
                text="It is a machine.",
            )
        ],
        style="bullet points",
        history=[{"question": "What is this?", "answer": "A device."}],
    )

    assert answer == "Answer"
    assert captured["model"] == "test-model"
    assert captured["messages"][0]["content"].endswith("bullet points")
    assert "User: What is this?" in captured["messages"][1]["content"]
    assert "Assistant: A device." in captured["messages"][1]["content"]


def test_rewrite_follow_up_uses_history(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": "What is the machine warranty?"},
                                )()
                            },
                        )()
                    ]
                },
            )()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("v2_app._llm_client", lambda: (FakeClient(), "test-model"))

    result = rewrite_follow_up(
        "And what about the warranty?",
        [{"question": "Tell me about the machine.", "answer": "It is an espresso machine."}],
    )

    assert result == "What is the machine warranty?"
    assert "It is an espresso machine." in captured["messages"][1]["content"]


def test_suggested_questions_use_retrieved_source_terms():
    citations = [
        Citation(
            document_id="own:manual.txt",
            source="manual.txt",
            chunk_id="manual.txt:0",
            text="Descaling removes limescale and improves heating efficiency.",
        )
    ]

    result = suggested_questions(citations)

    assert result
    assert all("documents say about" in question for question in result)
    assert any("limescale" in question for question in result)


def test_source_glossary_returns_source_sentence(tmp_path, monkeypatch):
    document = tmp_path / "manual.md"
    document.write_text("# Descaling\nDescaling removes limescale from the machine.", encoding="utf-8")
    monkeypatch.setattr(
        "v2_app.document_map",
        lambda: {
            "own:manual.md": {
                "id": "own:manual.md",
                "name": "manual.md",
                "path": document,
            }
        },
    )

    result = source_glossary(["own:manual.md"])

    assert result[0]["term"] == "Descaling"
    assert "limescale" in result[0]["definition"]
