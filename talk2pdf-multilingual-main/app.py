
import streamlit as st
from main import (
    get_pdf_text, get_text_chunks, get_vector_store,
    answer_question, gemini_tts, assemblyai_transcribe_bytes, 
    generate_quiz, VOICE_BY_LANG
)
import os

# -----------------------------
# Configuration & CSS
# -----------------------------
st.set_page_config(
    page_title="Talk2PDF Enhanced", 
    page_icon="📚", 
    layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("style.css")
except FileNotFoundError:
    pass



# -----------------------------
# Session State Initialization
# -----------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📚 Talk2PDF")
    st.markdown("---")
    
    st.subheader("1. Upload Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here", 
        accept_multiple_files=True,
        type=['pdf']
    )
    
    st.subheader("2. Settings")
    lang_options = {
        "English": "en", "Hindi": "hi", "Bengali": "bn", "Marathi": "mr",
        "Tamil": "ta", "Telugu": "te", "Spanish": "es", "French": "fr"
    }
    chosen_language = st.selectbox(
        "Output Language",
        options=list(lang_options.keys()),
        index=0
    )
    selected_lang_code = lang_options[chosen_language]
    
    st.markdown("---")
    process_button = st.button("🚀 Process Documents", use_container_width=True)

# -----------------------------
# Processing Logic
# -----------------------------
if process_button:
    if pdf_docs:
        with st.status("Processing your documents...", expanded=True) as status:
            st.write("Extracting text from PDFs...")
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text.strip():
                st.error("No text found in PDFs.")
                st.stop()
            
            st.session_state.raw_text = raw_text # Store for quiz
            
            st.write("Creating knowledge chunks...")
            chunks = get_text_chunks(raw_text)
            
            st.write("Building AI index...")
            get_vector_store(chunks)
            st.session_state.vector_store_ready = True
            st.session_state.chat_language = selected_lang_code
            status.update(label="Processing Complete!", state="complete", expanded=False)
            st.balloons()
    else:
        st.sidebar.error("Please upload at least one PDF.")

# -----------------------------
# Main Interface
# -----------------------------

# -----------------------------
# Main Interface
# -----------------------------
if not st.session_state.vector_store_ready:
    st.header("Talk2PDF Enhanced")
    st.write("The next generation of document interaction. Powered by Gemini 2.5.")
    st.info("👈 Upload your PDF documents in the sidebar to initialize the system.")

else:
    # Updates language if changed in sidebar after processing
    st.session_state.chat_language = selected_lang_code
    
    st.markdown("""
    <h2 style="text-align: center; margin-bottom: 30px; font-size: 2.5rem;">
        Workspace <span style="font-size: 1.5rem; opacity: 0.7;">// Active</span>
    </h2>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat Stream", "🗣️ Voice Interface", "🧠 Knowledge Check"])

    # --- TAB 1: CHAT ---
    with tab1:
        # Display chat history
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Input command or question..."):
            # Add user message
            st.session_state.conversation.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Processing neural response..."):
                    response = answer_question(prompt, st.session_state.chat_language)
                    st.write(response)
            
            # Add assistant message
            st.session_state.conversation.append({"role": "assistant", "content": response})

    # --- TAB 2: TALK ---
    with tab2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### 🎙️ Audio Input")
            audio_file = st.file_uploader("Upload Audio (wav/mp3)", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
        
        if audio_file:
            with st.spinner("Analysing audio waveforms..."):
                try:
                    query_text = assemblyai_transcribe_bytes(audio_file.read(), selected_lang_code)
                    st.success(f"Transcript: {query_text}")
                    
                    # Process as a question
                    with st.spinner("Computing answer..."):
                        answer_text = answer_question(query_text, st.session_state.chat_language)
                    
                    st.markdown(f"**Insight:** {answer_text}")
                    
                    # TTS
                    with st.spinner("Synthesizing voice output..."):
                        voice_name = VOICE_BY_LANG.get(st.session_state.chat_language, "kore")
                        audio_stream = gemini_tts(answer_text, voice_name)
                        st.audio(audio_stream, format="audio/wav")
                        
                except Exception as e:
                    st.error(f"Audio processing error: {e}")

    # --- TAB 3: QUIZ ---
    with tab3:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### 📝 Interactive Quiz")
        with c2:
            if st.button("🔄 Generate New"):
                with st.spinner("Analyzing content & generating questions..."):
                    if "raw_text" in st.session_state:
                        st.session_state.quiz_data = generate_quiz(st.session_state.raw_text, selected_lang_code)
                        st.session_state.quiz_answers = {} # Reset answers
                    else:
                        st.error("No context available.")

        if st.session_state.quiz_data:
            score = 0
            total = len(st.session_state.quiz_data)
            
            with st.form("quiz_form"):
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"#### Q{i+1}: {q['question']}")
                    
                    # Unique key for each question's radio button
                    selection = st.radio(
                        "Select an option:", 
                        q['options'], 
                        key=f"q_{i}",
                        index=None,
                        label_visibility="collapsed"
                    )
                    st.markdown("---")
                    
                submitted = st.form_submit_button("✅ Submit Answers")
                
                if submitted:
                    for i, q in enumerate(st.session_state.quiz_data):
                        user_val = st.session_state.get(f"q_{i}")
                        if user_val == q['answer']:
                            score += 1
                            st.success(f"Q{i+1}: Correct")
                        else:
                            st.error(f"Q{i+1}: Incorrect. Correct: {q['answer']}")
                    
                    st.balloons()
                    st.metric("Performance Score", f"{score}/{total}", delta=f"{int((score/total)*100)}%")
