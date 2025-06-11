# 🧠 AI Voice Financial Assistant

A Python-based AI assistant that understands Korean speech, searches a local financial knowledge base, and responds naturally using speech synthesis. It combines **OpenAI Whisper (STT)**, **GPT-4o (LLM)**, and **TTS API** to deliver a seamless voice-based experience.

---

## ✨ Key Features

- 🎤 **Voice-based Interaction**: Processes user queries from audio files.
- 🗣️ **Korean Language Support**: Recognizes Korean speech and responds in natural-sounding Korean voice.
- 📚 **Retrieval-Augmented Generation (RAG)**: Reduces hallucinations by answering based on .txt documents.
- 🧱 **Flexible Knowledge Base**: Expandable by adding more `.txt` files to the `documents/` folder.

---

## 📁 Project Structure
```.
├── documents/
│ ├── deposit_A.txt
│ └── savings_B.txt
├── .env
├── requirements.txt
├── build_db.py
├── financial_assistant_define.py
├── financial_assistant_live.ipynb
└── financial_assistant_record.ipynb
```

- `documents/`: Contains `.txt` files with financial product info (used as knowledge base).
- `.env`: Stores your OpenAI API key (`OPENAI_API_KEY="sk-..."`).
- `requirements.txt`: Python dependency list.
- `build_db.py`: Generates vector database (`financial_db.npy`, `financial_db_chunks.pkl`) from the documents.
- `financial_assistant.ipynb`: Main app – handles STT, RAG, LLM, and TTS flow.
- `financial_assistant_define.py`: Define modules and methods used in Main app

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone <your_repository_address>
cd <project_folder>
```

### 2. Set Environment Variables
Create a .env file in the root directory:

```bash
echo 'OPENAI_API_KEY="sk-..."' > .env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Financial Documents
Put .txt files (e.g., deposit_A.txt, savings_B.txt) into the documents/ folder. You can add more files to expand knowledge.

### 5. Build the Vector Database
Run this whenever you update the documents:

```bash
python build_db.py
```
-> Generates financial_db.npy and financial_db_chunks.pkl.

---

## ▶️ How to Run
This project works well in cloud IDEs like GitHub Codespaces.

1. Prepare Your Audio
Record a question (e.g., "청년에게 유리한 적금 상품 알려줘") as a .m4a, .mp3, .wav, etc.

2. Upload the Audio File
Drag-and-drop it into your Codespace file explorer.

3. Set File Name in Code
Open financial_assistant_record.ipynb and set your file:
```python
# In financial_assistant_record.ipynb
if __name__ == "__main__":
    INPUT_AUDIO_FILE = "your_audio_file.m4a"
    main(INPUT_AUDIO_FILE)
```

4. Run the Application
```bash
python financial_assistant_record.ipynb
```
5. Review the Results
- Terminal shows: Transcribed text (STT)
- AI-generated response (GPT-4o + RAG)
- Audio response saved as response.wav in the root folder.