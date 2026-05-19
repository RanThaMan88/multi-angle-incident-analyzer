import streamlit as st
import os
from pathlib import Path
import tempfile
from incident_analyzer import IncidentAnalyzer
import pandas as pd

st.set_page_config(page_title="Incident Analyzer", page_icon="🎥", layout="wide")

st.title("🎥 Multi-Angle Incident Analyzer")
st.markdown("Upload 2-3 videos of the same incident from different angles")

@st.cache_resource
def get_analyzer():
    return IncidentAnalyzer()

analyzer = get_analyzer()

tab1, tab2 = st.tabs(["📹 Upload & Analyze", "📊 Results"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Videos")
        case_name = st.text_input("Case Name", value="My Case")
        
        uploaded_files = {}
        st.subheader("Select Videos")
        for angle in ["bodycam", "dashcam", "incar"]:
            file = st.file_uploader(f"{angle.upper()}", type=["mp4", "avi", "mov"])
            if file:
                uploaded_files[angle] = file
    
    with col2:
        st.header("Status")
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} videos ready")
            size = sum(f.size for f in uploaded_files.values()) / (1024**3)
            st.metric("Total Size", f"{size:.2f} GB")
        else:
            st.info("📁 Upload videos to continue")
    
    st.divider()
    
    if st.button("▶️ START ANALYSIS", use_container_width=True, type="primary"):
        if len(uploaded_files) < 2:
            st.error("❌ Upload at least 2 videos")
        else:
            temp_dir = tempfile.mkdtemp()
            video_paths = {}
            
            with st.spinner("💾 Saving files..."):
                for angle, file in uploaded_files.items():
                    path = os.path.join(temp_dir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    video_paths[angle] = path
            
            with st.spinner("🔄 Analyzing (40+ minutes)..."):
                result = analyzer.analyze_incident(video_paths, case_name)
                st.session_state.result = result
            
            if result['success']:
                st.success("✅ Complete!")
            else:
                st.error(f"❌ Error: {result.get('error')}")

with tab2:
    st.header("📊 Results")
    
    if 'result' in st.session_state and st.session_state.result.get('success'):
        result = st.session_state.result
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Segments", len(result['transcript']))
        col2.metric("Cameras", len(result['sync_info']))
        col3.metric("Case", result['case_name'])
        
        st.subheader("📝 Transcript")
        data = []
        for seg in result['transcript']:
            data.append({
                'Time': f"{seg.start:.1f}s",
                'Speaker': seg.speaker,
                'Text': seg.text[:80] + "..." if len(seg.text) > 80 else seg.text
            })
        
        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        
        st.subheader("🔄 Sync Info")
        sync_data = []
        for angle, info in result['sync_info'].items():
            sync_data.append({
                'Camera': angle.upper(),
                'Offset': f"{info['offset']:.2f}s",
                'Status': '✅ Synced'
            })
        
        st.dataframe(pd.DataFrame(sync_data), use_container_width=True)
    else:
        st.info("Upload and analyze to see results")