import gradio as gr
from transformers import pipeline
from gtts import gTTS
from deep_translator import GoogleTranslator
import os
import nltk
from nltk.corpus import stopwords
import tempfile 
import torch # Used for device mapping

# --- CONFIG & SETUP ---
device_id = 0 if torch.cuda.is_available() else -1

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Dictionaries (Minimal)
TRANSLATION_LANGUAGES = {
    "None": None, "French": "fr", "Spanish": "es", "Hindi": "hi", "English": "en" 
}
EMOJI_MAPPING = {
    "joy": "😃", "anger": "😡", "sadness": "😢", "fear": "😨", "love": "😍", "surprise": "😲", "disgust": "🤢", "neutral": "😐"
}
ASL_DICTIONARY = {
    "hello": "https://www.lifeprint.com/asl101/gifs/h/hello.gif",
    "thank": "https://www.lifeprint.com/asl101/gifs/t/thank-you.gif",
    "love": "https://www.lifeprint.com/asl101/gifs/l/love.gif",
    "happy": "https://www.lifeprint.com/asl101/gifs/h/happy.gif",
}
ASL_ALPHABET_IMAGES = {letter: f"https://www.lifeprint.com/asl101/fingerspelling/{letter}.gif" for letter in "abcdefghijklmnopqrstuvwxyz"}

# Load Models onto GPU/CPU
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_id) 
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=device_id)

# --- UTILITY FUNCTIONS ---

def get_sign_language_representation(text):
    words = text.lower().split()
    images_html = """<div style='overflow-x: auto; white-space: nowrap; display: flex;'>"""
    for word in words:
        clean_word = ''.join(filter(str.isalpha, word))
        if clean_word in ASL_DICTIONARY:
            images_html += f"<img src='{ASL_DICTIONARY[clean_word]}' style='width: 50px; height: 50px; margin-right: 5px;'>"
        else:
            for char in clean_word:
                if char in ASL_ALPHABET_IMAGES:
                    images_html += f"<img src='{ASL_ALPHABET_IMAGES[char]}' style='width: 50px; height: 50px; margin-right: 2px;'>"
    return images_html + "</div>"

def translate_text(text, target_language_name):
    lang_code = TRANSLATION_LANGUAGES.get(target_language_name)
    if lang_code:
        return GoogleTranslator(source="auto", target=lang_code).translate(text)
    return text

def text_to_speech(text, target_language_name):
    lang_code = TRANSLATION_LANGUAGES.get(target_language_name, "en")
    try:
        if not text: return None
        with tempfile.NamedTemporaryFile(suffix=f"_{lang_code}.mp3", delete=False) as fp:
            gTTS(text, lang=lang_code, slow=False).save(fp.name)
            return fp.name
    except Exception:
        return None

# --- MAIN FUNCTION (fn) ---

def summarize_text(text, target_language, min_words=30, max_words=150):
    if not text.strip():
        return "Input required.", None, None, "N/A", "N/A", ""

    # 1. Summarization (Optimized with num_beams=2)
    summary_params = {'max_length': int(max_words), 'min_length': int(min_words), 'do_sample': False, 'num_beams': 2}
    summary = summarizer(text, **summary_params)[0]['summary_text']
    
    # 2. Emotion Detection
    emotion = emotion_detector(summary)[0]['label']
    emoji_display = EMOJI_MAPPING.get(emotion, "😐")

    # 3. Translation
    english_summary = translate_text(summary, "English")
    translated_summary = translate_text(summary, target_language) 
    
    # 4. Audio & ASL Generation
    audio_path_en = text_to_speech(english_summary, "English")
    audio_path_target = None
    if target_language and target_language != "None" and translated_summary:
        audio_path_target = text_to_speech(translated_summary, target_language)
        
    asl_images_html = get_sign_language_representation(summary)

    # Return 6 outputs
    return (
        f"{summary} \n\nEmotion: {emotion} {emoji_display}", # 1
        audio_path_en,                                      # 2
        audio_path_target,                                  # 3
        english_summary,                                    # 4
        translated_summary,                                 # 5
        asl_images_html                                     # 6
    )

# --- GRADIO INTERFACE ---

demo = gr.Interface(
    fn=summarize_text,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter text to summarize..."),
        gr.Dropdown(choices=list(TRANSLATION_LANGUAGES.keys()), value="None", label="Translate to (Optional)"),
        gr.Slider(minimum=10, maximum=100, value=30, label="Min Words"),
        gr.Slider(minimum=50, maximum=300, value=150, label="Max Words")
    ],
    outputs=[
        gr.Textbox(label="Summary & Emotion"),
        gr.Audio(label="English Audio"),             
        gr.Audio(label="Selected Language Audio"),   
        gr.Textbox(label="English Text"),
        gr.Textbox(label="Selected Language Text"),
        gr.HTML(label="ASL Representation"),
    ],
    title="AI Summarizer with Dual Audio & ASL (Optimized)",
    description="Summarize, detect emotion, translate, and generate dual audio/ASL. **Requires GPU hardware accelerator on Hugging Face Spaces for fast results.**"
)

if __name__ == "__main__":
    demo.launch()
