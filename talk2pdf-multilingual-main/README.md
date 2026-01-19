# 📚 Talk2PDF Enhanced

**Interact with your PDF documents like never before.**  
Chat, Talk, and Quiz yourself on your document's content in multiple languages.

## 🚀 Key Features

- **💬 Chat Mode**: Ask questions and get instant, context-aware answers from your PDFs using RAG (Retrieval Augmented Generation).
- **🗣️ Voice Interface**: Hands-free interaction.
    - **Speech-to-Text**: Speak your queries using AssemblyAI's advanced transcription.
    - **Text-to-Speech**: Listen to AI-synthesized responses powered by Gemini 2.5 Flash TTS.
- **🧠 Quiz Mode**: Automatically generate interactive 5-question multiple-choice quizzes to test your retention.
- **🌍 Multilingual Support**: seamless interaction in **English, Hindi, Bengali, Marathi, Tamil, Telugu, Spanish, and French**.
- **🎨 2026 Tech UI**: A futuristic, clean interface featuring 'Space Grotesk' typography and 'Inter' body text.

## 🛠️ Tech Stack

This project leverages the latest in Generative AI and web technologies:

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: Google **Gemini 2.5 Flash** (via `google-genai` and `langchain-google-genai`)
- **Embeddings**: Google Generative AI Embeddings (`models/gemini-embedding-001`)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Framework**: [LangChain](https://www.langchain.com/) for RAG pipeline orchestration.
- **Audio Processing**: 
    - **STT**: [AssemblyAI](https://www.assemblyai.com/)git commit -m "first commit"
    - **TTS**: Gemini 2.5 Flash Preview TTS
- **PDF Parsing**: PyPDF2

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd talk2pdf-multilingual-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys**
   - Create a `.env` file in the root directory.
   - Add your API keys:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
     ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 📝 Usage

1. **Upload**: Upload one or multiple PDF files in the sidebar.
2. **Process**: Click "🚀 Process Documents" to index the content.
3. **Select Language**: Choose your preferred output language from the sidebar dropdown.
4. **Interact**:
   - **Chat Stream**: Type questions to chat with your document.
   - **Voice Interface**: Upload an audio question (.wav/.mp3) to talk to your document.
   - **Knowledge Check**: Click "Generate New" to create a quiz based on the PDF content.

