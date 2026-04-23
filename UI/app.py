import streamlit as st
import requests
import json

# ===== CONFIG =====
API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Text Processor", layout="wide")

# ===== TITLE =====
st.title("🧠 AI Text Processing (VI ↔ EN)")
st.markdown("Tóm tắt • Dịch • Nhận diện thực thể")

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Settings")

source_lang = st.sidebar.selectbox(
    "Ngôn ngữ đầu vào",
    ["vi", "en"]
)

target_lang = st.sidebar.selectbox(
    "Ngôn ngữ dịch sang",
    ["en", "vi"]
)

api_url = st.sidebar.text_input("API URL", API_URL)

# ===== INPUT =====
st.subheader("📥 Nhập văn bản")

text_input = st.text_area(
    "Nhập text (Vietnamese hoặc English)",
    height=250
)

process_btn = st.button("🚀 Process")

# ===== PROCESS =====
if process_btn:
    if not text_input.strip():
        st.warning("⚠️ Vui lòng nhập văn bản")
    else:
        with st.spinner("Đang xử lý..."):

            try:
                response = requests.post(
                    f"{api_url}/api/process",
                    json={
                        "text": text_input,
                        "source_lang": source_lang,
                        "target_lang": target_lang
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("✔ Xử lý thành công!")

                    # ===== DISPLAY =====
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📄 Văn bản gốc")
                        st.write(result.get("original_text_vi", text_input))

                        st.subheader("📝 Summary (VI)")
                        st.write(result.get("summary_vi", ""))

                    with col2:
                        st.subheader("🌍 Bản dịch")
                        st.write(result.get("translated_text_en", ""))

                        st.subheader("📝 Summary (EN)")
                        st.write(result.get("summary_en", ""))

                    # ===== ENTITIES =====
                    st.subheader("🏷️ Entities")

                    entities = result.get("entities", [])

                    if entities:
                        for ent in entities:
                            st.write(f"- {ent['text']} ({ent['type']})")
                    else:
                        st.write("Không có entities")

                    # ===== DOWNLOAD =====
                    st.subheader("⬇️ Download")

                    json_data = json.dumps(result, ensure_ascii=False, indent=2)

                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="result.json",
                        mime="application/json"
                    )

                else:
                    st.error(f"❌ API lỗi: {response.text}")

            except Exception as e:
                st.error(f"❌ Lỗi kết nối API: {e}")
