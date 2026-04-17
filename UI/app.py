import json

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"

st.set_page_config(page_title="Mô hình NLP", layout="wide")

st.title("Mô hình Dịch thuật và Tóm tắt Văn bản lớn")
st.caption("Transformer Seq2Seq cho tóm tắt, kèm graph quan hệ nhân vật.")

st.sidebar.header("Cấu hình")
source_lang = st.sidebar.selectbox("Ngôn ngữ đầu vào", ["vi", "en"], index=0)
target_lang = st.sidebar.selectbox("Ngôn ngữ đích đến", ["en", "vi"], index=0)
api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL)

st.subheader("Văn bản đầu vào")
text_input = st.text_area(
    "Nhập nội dung truyện, chương truyện hoặc văn bản dài",
    height=280,
    placeholder="Ví dụ: Đoạn văn 10,000 chữ về trận chiến giữa nhân vật A và nhân vật B...",
)

if st.button("Xử lý"):
    if not text_input.strip():
        st.warning("Vui lòng nhập văn bản trước khi xử lý.")
    else:
        payload = {
            "text": text_input,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        try:
            with st.spinner("Đang gọi mô hình..."):
                response = requests.post(f"{api_url}/api/process", json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            st.error(f"Không thể gọi API: {exc}")
        else:
            st.success("Xử lý thành công.")
            metadata = result.get("metadata", {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Văn bản gốc**")
                st.write(result.get("original_text", ""))

                summary_vi_original = result.get("summary_vi_original", "")
                summary_vi_active = result.get("summary_vi", "")

                if summary_vi_original:
                    st.markdown("**Tóm tắt tiếng Việt (Bản Gốc)**")
                    st.write(summary_vi_original)

                if not summary_vi_original and summary_vi_active:
                    st.markdown("**Tóm tắt tiếng Việt**")
                    st.write(summary_vi_active)

            with col2:
                st.markdown("**Bản dịch**")
                st.write(result.get("translated_text", ""))

                summary_en_original = result.get("summary_en_original", "")
                summary_en_active = result.get("summary_en", "")

                if summary_en_original:
                    st.markdown("**Tóm tắt tiếng Anh (Bản Gốc)**")
                    st.write(summary_en_original)
                elif summary_en_active:
                    st.markdown("**Tóm tắt tiếng Anh**")
                    st.write(summary_en_active)

            st.subheader("Graph quan hệ nhân vật")
            relation_graph = result.get("relation_graph", {})
            nodes = relation_graph.get("nodes", [])
            edges = relation_graph.get("edges", [])

            if nodes:
                st.markdown("**Danh sách nhân vật**")
                st.dataframe(pd.DataFrame(nodes), use_container_width=True)
            else:
                st.info("Chưa tìm thấy nhân vật rõ ràng trong văn bản đã tóm tắt.")

            if edges:
                st.markdown("**Danh sách quan hệ**")
                st.dataframe(pd.DataFrame(edges), use_container_width=True)
            else:
                st.info("Chưa tìm thấy cạnh quan hệ, graph hiện tại chỉ có node hoặc văn bản quá ngắn.")

            st.subheader("Thông tin bổ sung")
            st.json(metadata, expanded=False)

            st.subheader("Tải kết quả")
            json_data = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="nlp_result.json",
                mime="application/json",
            )
