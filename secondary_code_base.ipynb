!pip install numpy numba
!pip install git+https://github.com/openai/whisper.git --no-deps
!pip install gradio librosa matplotlib fpdf
!pip install tiktoken

import gradio as gr
import whisper
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import tempfile



model = whisper.load_model("base")

analysis_data = []

def analyze_audio(file):
    # Load audio
    y, sr = librosa.load(file, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = np.sqrt(np.mean(y**2))
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    avg_pitch = np.mean(pitch)
    
    # Transcription
    result = model.transcribe(file)
    transcript = result["text"]

    # Save data
    metrics = {
        "Filename": os.path.basename(file),
        "Duration (s)": round(duration, 2),
        "Loudness (RMS)": round(rms, 4),
        "Avg Pitch (Hz)": round(avg_pitch, 2),
        "Transcript": transcript.strip()
    }
    analysis_data.append(metrics)

    return transcript, pd.DataFrame([metrics])


def download_csv():
    df = pd.DataFrame(analysis_data)
    path = "/tmp/voice_analysis_data.csv"
    df.to_csv(path, index=False)
    return path

def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for row in analysis_data:
        pdf.cell(200, 10, txt=f"File: {row['Filename']}", ln=True)
        pdf.cell(200, 10, txt=f"Duration: {row['Duration (s)']}s", ln=True)
        pdf.cell(200, 10, txt=f"Loudness: {row['Loudness (RMS)']}", ln=True)
        pdf.cell(200, 10, txt=f"Avg Pitch: {row['Avg Pitch (Hz)']} Hz", ln=True)
        pdf.multi_cell(0, 10, txt=f"Transcript: {row['Transcript']}")
        pdf.ln(10)

    report_path = "/tmp/voice_report.pdf"
    pdf.output(report_path)
    return report_path

def plot_metrics():
    if not analysis_data:
        return None
    
    df = pd.DataFrame(analysis_data)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df.plot(x="Filename", y="Duration (s)", kind="bar", ax=ax[0], legend=False, color='skyblue')
    ax[0].set_title("Audio Duration")

    df.plot(x="Filename", y="Avg Pitch (Hz)", kind="bar", ax=ax[1], legend=False, color='orange')
    ax[1].set_title("Average Pitch")

    plt.tight_layout()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name)
    return temp_file.name

with gr.Blocks() as app:
    gr.Markdown("## 🎙️ Voice Analysis App")

    with gr.Tab("Analyze"):
        audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
        transcribed = gr.Textbox(label="Transcript")
        metrics = gr.Dataframe(headers=["Filename", "Duration (s)", "Loudness (RMS)", "Avg Pitch (Hz)", "Transcript"])
        analyze_btn = gr.Button("Analyze Audio")
        download_btn = gr.Button("⬇️ Download CSV")
        report_btn = gr.Button("📄 Generate PDF Report")
        csv_out = gr.File(label="CSV File")
        pdf_out = gr.File(label="PDF Report")

        analyze_btn.click(analyze_audio, inputs=audio_input, outputs=[transcribed, metrics])
        download_btn.click(download_csv, outputs=csv_out)
        report_btn.click(generate_report, outputs=pdf_out)

    with gr.Tab("Compare Metrics"):
        compare_btn = gr.Button("📊 Show Comparison Plots")
        image_output = gr.Image()
        compare_btn.click(plot_metrics, outputs=image_output)

app.launch()
