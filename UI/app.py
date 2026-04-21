import json

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"

st.set_page_config(page_title="NLP • Translation & Summarization", layout="wide")

# ---------- UI styling (no feature changes) ----------
st.markdown(
    """
    <style>
      /* Keep content centered and readable */
      .block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; max-width: 1200px; }
      /* Buttons full width in sidebar */
      section[data-testid="stSidebar"] button { width: 100%; }
      /* Make text areas look nicer */
      textarea { font-size: 0.95rem !important; line-height: 1.35rem !important; }
      /* Slightly reduce default header spacing */
      h1, h2, h3 { margin-bottom: 0.6rem; }
      /* Nice “card” look for containers with border */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        border-color: rgba(49, 51, 63, 0.12);
      }
      /* Dataframe a bit tighter */
      .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.title("NLP Translation & Summarization")
st.caption("Dịch thuật • Tóm tắt văn bản lớn • Graph quan hệ nhân vật")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Cấu hình")

    source_lang = st.selectbox("Ngôn ngữ đầu vào", ["vi", "en"], index=0)
    target_lang = st.selectbox("Ngôn ngữ đích đến", ["en", "vi"], index=0)
    api_url = st.text_input("API URL", DEFAULT_API_URL)

    st.divider()
    st.subheader("Gợi ý")
    st.write("- Nhập văn bản dài để thấy chế độ tóm tắt phân cấp (nếu có).")
    st.write("- Có thể tải JSON kết quả ở cuối.")

# ---------- Input ----------
st.subheader("Văn bản đầu vào")
with st.container(border=True):
    text_input = st.text_area(
        "Nhập nội dung truyện, chương truyện hoặc văn bản dài",
        height=260,
        placeholder="Ví dụ: Đoạn văn dài về trận chiến giữa nhân vật A và nhân vật B...",
        label_visibility="collapsed",
    )
    run = st.button("Xử lý", type="primary")

# ---------- Process ----------
if run:
    if not text_input.strip():
        st.warning("Vui lòng nhập văn bản trước khi xử lý.")
    else:
        payload = {
            "text": text_input,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        try:
            with st.spinner("Đang gọi API /api/process ..."):
                response = requests.post(f"{api_url}/api/process", json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            st.error(f"Không thể gọi API: {exc}")
        else:
            st.success("Xử lý thành công.")

            # Extract for presentation only (no logic change)
            metadata = result.get("metadata", {}) or {}
            src_meta = (metadata.get("source_summary") or {}) if isinstance(metadata, dict) else {}
            tgt_meta = (metadata.get("target_summary") or {}) if isinstance(metadata, dict) else {}

            # Top quick metrics (pure UI)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Source", source_lang.upper())
            with c2:
                st.metric("Target", target_lang.upper())
            with c3:
                st.metric("Input tokens", src_meta.get("input_tokens", "—"))
            with c4:
                st.metric("Summary mode", metadata.get("summary_mode", "—"))

            st.divider()

            # Tabs to keep layout clean
            tab_sum, tab_tr, tab_graph, tab_meta, tab_dl = st.tabs(
                ["Tóm tắt", "Dịch", "Graph", "Metadata", "Tải JSON"]
            )

            with tab_sum:
                left, right = st.columns(2)
                with left:
                    st.markdown("#### Văn bản gốc")
                    with st.container(border=True):
                        st.write(result.get("original_text", ""))

                with right:
                    st.markdown("#### Tóm tắt")
                    with st.container(border=True):
                        # Keep exactly existing behavior: show whichever summary fields exist
                        summary_vi_original = result.get("summary_vi_original", "")
                        summary_vi_active = result.get("summary_vi", "")
                        summary_en_original = result.get("summary_en_original", "")
                        summary_en_active = result.get("summary_en", "")

                        if summary_vi_original:
                            st.markdown("**Tóm tắt tiếng Việt (Bản gốc)**")
                            st.write(summary_vi_original)
                        elif summary_vi_active:
                            st.markdown("**Tóm tắt tiếng Việt**")
                            st.write(summary_vi_active)

                        st.markdown("---")

                        if summary_en_original:
                            st.markdown("**Tóm tắt tiếng Anh (Bản gốc)**")
                            st.write(summary_en_original)
                        elif summary_en_active:
                            st.markdown("**Tóm tắt tiếng Anh**")
                            st.write(summary_en_active)

            with tab_tr:
                st.markdown("#### Bản dịch")
                with st.container(border=True):
                    st.write(result.get("translated_text", ""))

                note = (metadata.get("translation_note") if isinstance(metadata, dict) else None) or ""
                if note:
                    st.caption(f"Ghi chú: {note}")

            with tab_graph:
                st.markdown("#### Graph quan hệ nhân vật")
                relation_graph = result.get("relation_graph", {}) or {}
                nodes = relation_graph.get("nodes", []) or []
                edges = relation_graph.get("edges", []) or []

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Nodes**")
                    with st.container(border=True):
                        if nodes:
                            st.dataframe(pd.DataFrame(nodes), use_container_width=True, height=360)
                        else:
                            st.info("Chưa tìm thấy nhân vật rõ ràng trong văn bản.")

                with col2:
                    st.markdown("**Edges**")
                    with st.container(border=True):
                        if edges:
                            st.dataframe(pd.DataFrame(edges), use_container_width=True, height=360)
                        else:
                            st.info("Chưa tìm thấy cạnh quan hệ (graph có thể chỉ có node hoặc văn bản quá ngắn).")

            with tab_meta:
                st.markdown("#### Thông tin bổ sung")
                with st.container(border=True):
                    st.json(metadata, expanded=False)

            with tab_dl:
                st.markdown("#### Tải kết quả")
                with st.container(border=True):
                    json_data = json.dumps(result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="nlp_result.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    with st.expander("Xem trước JSON"):
                        st.code(json_data, language="json")