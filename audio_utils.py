import os
import torch
import librosa
import numpy as np
from faster_whisper import WhisperModel
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification, pipeline
import soundfile as sf

# Initialize models globally to avoid reloading (Streamlit caching should be used in app.py really, but for clean separation let's define classes/funcs)

class AudioProcessor:
    def __init__(self):
        # ASR Model
        print("Loading Whisper model...")
        self.asr_model = WhisperModel("small", device="cpu", compute_type="int8") # Use base/cpu for speed
        
        # Emotion Model
        print("Loading Hubert model...")
        self.emotion_model_name = "superb/hubert-base-superb-er"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.emotion_model_name)
        self.emotion_model = HubertForSequenceClassification.from_pretrained(self.emotion_model_name)
        
        # Emotion Labels (from hubert-base-superb-er config)
        # usually: neu, hap, ang, sad
        self.id2label = self.emotion_model.config.id2label
        
        # Text Translation & Emotion Models (Lazy Init)
        self.translator = None
        self.text_emotion_classifier = None

    def transcribe(self, audio_path):
        """
        Transcribes audio file to text.
        """
        segments, info = self.asr_model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text

    def predict_emotion(self, audio_path):
        """
        Predicts emotion from audio file.
        """
        # Load audio at 16k for Hubert
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Process
        inputs = self.feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = self.emotion_model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_label = self.id2label[predicted_ids.item()]
        
        return predicted_label

    def predict_text_emotion(self, text_ja):
        """
        Predicts emotion from Japanese text (translates to EN first).
        Returns mapped label: 'neu', 'hap', 'sad', 'ang'
        """

        if not text_ja:
            return 'neu'
            
        # Lazy Load
        if self.translator is None or self.text_emotion_classifier is None:
            print("Loading Text Translation & Emotion models (Lazy)...")
            try:
                self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
                self.text_emotion_classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", top_k=1)
            except Exception as e:
                print(f"Warning: Failed to load text models: {e}")
                return 'neu'

        if not self.translator or not self.text_emotion_classifier:
             return 'neu'
            
        try:
            # 1. Translate JA -> EN
            translation_result = self.translator(text_ja)
            text_en = translation_result[0]['translation_text']
            print(f"Translated: {text_en}")
            
            # 2. Predict Emotion (EN)
            # labels: joy, optimism, anger, sadness, fear, surprise
            results = self.text_emotion_classifier(text_en)
            top_result = results[0][0] # top_k=1 returns list of lists
            label = top_result['label']
            score = top_result['score']
            
            print(f"Text Emotion Raw: {label} ({score})")
            
            # 3. Map to standard labels
            # Standard: neu, hap, sad, ang
            label_map = {
                'joy': 'hap',
                'optimism': 'hap',
                'anger': 'ang',
                'sadness': 'sad',
                'fear': 'sad', # Map fear to sad or ang? sad for now
                'surprise': 'hap', # Map surprise to hap?
                'neutral': 'neu'   # If model has neutral
            }
            
            return label_map.get(label, 'neu')
            
        except Exception as e:
            print(f"Text Emotion Error: {e}")
            return 'neu'

def save_audio_bytes(audio_bytes, file_path="temp_audio.wav"):
    """
    Saves bytes to a wav file.
    """
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path
