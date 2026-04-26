from relation_graph import build_relation_graph


def to_entities(graph: dict) -> list[dict]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    return [
        {
            "text": node.get("label", ""),
            "type": "CHARACTER",
            "mentions": node.get("mentions", 0),
        }
        for node in nodes
    ]


def test_empty_text_returns_empty_list():
    graph = build_relation_graph("")
    assert to_entities(graph) == []
    graph = build_relation_graph("   \n\t  ")
    assert to_entities(graph) == []


def test_output_format_is_list_of_dicts():
    text = "Ong Nguyen Van A song o Ha Noi va lam viec tai VNPT."
    ents = to_entities(build_relation_graph(text))

    assert isinstance(ents, list)
    for e in ents:
        assert isinstance(e, dict)
        assert "text" in e and isinstance(e["text"], str)
        assert "type" in e and isinstance(e["type"], str)
        assert "mentions" in e and isinstance(e["mentions"], int)
        assert e["text"].strip() != ""
        assert e["type"].strip() != ""


def test_contains_some_entities():
    text = "Ong Nguyen Van A song o Ha Noi."
    ents = to_entities(build_relation_graph(text))
    assert len(ents) >= 1
