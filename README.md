# Voice-Cognition-Analysis-Final
An interactive voice cognition analysis tool built in Google Colab with Whisper, Librosa, and Gradio. It allows users to upload audio files, trim sections, analyze voice features, and detect anomalies or cluster patterns using unsupervised learning.
## 📦 Features

-  Audio upload and trimming
-  Transcription using OpenAI Whisper
-  Feature extraction:
  - Pause count
  - Filler word ratio
  - Speech rate
  - Pitch variability
  - Sentiment score
  - Energy
- 🧠 KMeans clustering + Isolation Forest anomaly detection
- 📈 Visual analysis (Seaborn)
- 📄 PDF report and CSV export
- 🌐 Easy-to-use Gradio UI

---

## 🚀 How to Run

1. Open the Colab notebook.
2. Install requirements:
   ```bash
   !pip install -q gradio pandas matplotlib seaborn fpdf librosa transformers tts git+https://github.com/openai/whisper.git
