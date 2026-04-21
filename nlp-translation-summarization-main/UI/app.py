"""
Streamlit UI for NLP Translation & Summarization API
"""

import json
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 180  # 3 minutes

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Mô hình NLP",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Mô hình Dịch thuật & Tóm tắt Văn bản")
st.caption("Transformer Seq2Seq: Tóm tắt + Dịch + Graph quan hệ nhân vật")

# ==================== SIDEBAR CONFIG ====================

st.sidebar.header("⚙️ Cấu hình")

source_lang = st.sidebar.selectbox(
    "📖 Ngôn ngữ đầu vào",
    ["vi", "en"],
    index=0,
    help="Chọn ngôn ngữ của văn bản đầu vào"
)

target_lang = st.sidebar.selectbox(
    "🎯 Ngôn ngữ đích",
    ["en", "vi"],
    index=0,
    help="Chọn ngôn ngữ cần dịch tới"
)

api_url = st.sidebar.text_input(
    "🔗 API URL",
    value=DEFAULT_API_URL,
    help="Địa chỉ server FastAPI"
)

# Advanced settings (collapsed)
with st.sidebar.expander("⚙️ Cài đặt nâng cao"):
    max_input_length = st.number_input(
        "Max input length",
        min_value=128,
        max_value=2048,
        value=1024,
        step=128
    )
    max_new_tokens = st.number_input(
        "Max output tokens",
        min_value=32,
        max_value=1024,
        value=512,
        step=32
    )
    num_beams = st.number_input(
        "Number of beams",
        min_value=1,
        max_value=8,
        value=4
    )

# ==================== INPUT SECTION ====================

st.subheader("📝 Văn bản đầu vào")

text_input = st.text_area(
    "Nhập nội dung truyện, chương truyện hoặc văn bản dài",
    height=250,
    placeholder="Ví dụ: Đoạn văn 10,000 chữ về trận chiến giữa nhân vật A và nhân vật B...",
    help="Hỗ trợ tối đa 2048 ký tự"
)

# ==================== PROCESSING ====================

col1, col2, col3 = st.columns([2, 1, 1])

with col2:
    process_button = st.button("🚀 Xử lý", use_container_width=True)

with col3:
    clear_button = st.button("🗑️ Xóa", use_container_width=True)

if clear_button:
    st.session_state.clear()
    st.rerun()

if process_button:
    if not text_input.strip():
        st.warning("⚠️ Vui lòng nhập văn bản trước khi xử lý.")
    else:
        # Prepare payload
        payload = {
            "text": text_input,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "max_input_length": max_input_length,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
        }
        
        try:
            with st.spinner("⏳ Đang xử lý... (điều này có thể mất vài phút)"):
                response = requests.post(
                    f"{api_url}/api/process",
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
            
            # Check response status
            response.raise_for_status()
            result = response.json()
            
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Không thể kết nối tới API: {api_url}")
            st.info("💡 Hãy chắc chắn rằng server đang chạy: `python server.py`")
        
        except requests.exceptions.Timeout:
            st.error(f"⏱️ Hết thời gian chờ ({REQUEST_TIMEOUT}s). Văn bản quá dài hoặc server bị quá tải.")
        
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Lỗi API: {response.status_code}")
            st.error(response.text)
        
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
        
        else:
            # Success
            st.success("✅ Xử lý thành công!")
            
            # ==================== RESULTS ====================
            
            # Extract metadata
            metadata = result.get("metadata", {})
            
            # Display summaries and translations side by side
            st.subheader("📋 Kết quả")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📖 Văn bản gốc")
                st.write(result.get("original_text", ""))
                
                summary_vi_original = result.get("summary_vi_original", "")
                summary_vi_active = result.get("summary_vi", "")
                
                if summary_vi_original:
                    st.markdown("### 📝 Tóm tắt Tiếng Việt")
                    st.write(summary_vi_original)
                elif summary_vi_active:
                    st.markdown("### 📝 Tóm tắt Tiếng Việt")
                    st.write(summary_vi_active)
            
            with col2:
                st.markdown("### 🌐 Bản dịch")
                translated_text = result.get("translated_text", "")
                if translated_text:
                    st.write(translated_text)
                else:
                    st.info("Không có bản dịch")
                
                summary_en_original = result.get("summary_en_original", "")
                summary_en_active = result.get("summary_en", "")
                
                if summary_en_original:
                    st.markdown("### 📝 Tóm tắt Tiếng Anh")
                    st.write(summary_en_original)
                elif summary_en_active:
                    st.markdown("### 📝 Tóm tắt Tiếng Anh")
                    st.write(summary_en_active)
            
            # ==================== RELATION GRAPH ====================
            
            st.subheader("🔗 Graph quan hệ nhân vật")
            
            relation_graph = result.get("relation_graph", {})
            nodes = relation_graph.get("nodes", [])
            edges = relation_graph.get("edges", [])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if nodes:
                    st.markdown("**👥 Danh sách nhân vật**")
                    nodes_df = pd.DataFrame(nodes)
                    st.dataframe(nodes_df, use_container_width=True, hide_index=True)
                else:
                    st.info("📭 Chưa tìm thấy nhân vật rõ ràng")
            
            with col2:
                if edges:
                    st.markdown("**🔗 Danh sách quan hệ**")
                    edges_df = pd.DataFrame(edges)
                    st.dataframe(edges_df, use_container_width=True, hide_index=True)
                else:
                    st.info("📭 Chưa tìm thấy quan hệ giữa nhân vật")
            
            # ==================== METADATA ====================
            
            st.subheader("📊 Thông tin bổ sung")
            
            with st.expander("📈 Chi tiết xử lý", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "💾 Tokens (input)",
                        metadata.get("source_summary", {}).get("input_tokens", "N/A")
                    )
                
                with col2:
                    st.metric(
                        "📤 Chunks",
                        metadata.get("source_summary", {}).get("chunk_count", "N/A")
                    )
                
                with col3:
                    st.metric(
                        "🔀 Mode",
                        "Hierarchical" if metadata.get("source_summary", {}).get("auto_hierarchical") else "Direct"
                    )
                
                st.json(metadata, expanded=False)
            
            # ==================== DOWNLOAD ====================
            
            st.subheader("💾 Tải kết quả")
            
            json_data = json.dumps(result, ensure_ascii=False, indent=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="nlp_result.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # CSV export (summaries only)
                csv_data = f"""Original Text,Summary VI,Summary EN,Translated Text
"{result.get('original_text', '')}","{summary_vi_original or summary_vi_active}","{summary_en_original or summary_en_active}","{translated_text}"
"""
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name="nlp_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==================== FOOTER ====================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔧 FastAPI Server: http://localhost:8000")

with col2:
    st.caption("📚 Docs: http://localhost:8000/docs")

with col3:
    st.caption("💻 GitHub: TranNgocDong/nlp-translation-summarization")