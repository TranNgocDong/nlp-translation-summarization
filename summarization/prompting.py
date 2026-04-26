from __future__ import annotations


def wrap_vi_like_training(text_vi: str) -> str:
    t = (text_vi or "").strip()
    if not t:
        return t
        
    try:
        from relation_graph import build_relation_graph
        graph = build_relation_graph(t)
        # Lấy top 7 entity xuất hiện nhiều nhất để không làm trôi prompt
        entities = [str(node["id"]) for node in graph.get("nodes", [])][:7]
        ner_str = ", ".join(entities) if entities else ""
    except Exception as e:
        ner_str = ""
        
    prefix = f"Từ khóa bắt buộc: {ner_str}\n" if ner_str else ""

    if ("Tiêu đề:" in t) or ("Nội dung:" in t) or ("Tóm tắt nhanh:" in t):
        return f"{prefix}{t}"
    return f"{prefix}Nội dung:\n{t}"

