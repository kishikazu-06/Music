# app.py
import os
import io
import wave
from typing import Optional, List, Dict
from dataclasses import dataclass

import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

import librosa
import torch
from faster_whisper import WhisperModel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

# --------------------------
# 定数
# --------------------------
SAMPLE_RATE = 16000
MIN_AUDIO_LEN = SAMPLE_RATE
WHISPER_SIZE = "small"

SPOTIPY_CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")

EMOTION_KEYWORD_MAP = {
    "anger": "angry 怒り",
    "sadness": "sad 悲しい",
    "joy": "happy 楽しい",
    "optimism": "positive ポジティブ",
    "neutral": "calm neutral",
    "surprise": "surprise",
    "love": "love",
}

# --------------------------
# ユーティリティ
# --------------------------
def force_min_length(waveform: np.ndarray, target_length: int = MIN_AUDIO_LEN) -> np.ndarray:
    if waveform is None:
        return np.zeros(target_length, dtype=np.float32)
    length = waveform.shape[0]
    if length >= target_length:
        return waveform
    pad = target_length - length
    return np.concatenate([waveform, np.zeros(pad, dtype=np.float32)])

def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()

# --------------------------
# モデルロード
# --------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    with st.spinner("モデルをロード中..."):
        audio_fe = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
        audio_model = AutoModelForAudioClassification.from_pretrained("superb/hubert-base-superb-er")

        text_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        text_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

        whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    return {
        "audio_feature_extractor": audio_fe,
        "audio_model": audio_model,
        "text_tokenizer": text_tok,
        "text_model": text_model,
        "whisper_model": whisper,
    }

# --------------------------
# Spotify
# --------------------------
@st.cache_resource
def get_spotify_client():
    if not SPOTIPY_CLIENT_ID:
        return None
    auth = SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
    )
    return spotipy.Spotify(auth_manager=auth)

# --------------------------
# 推論関数
# --------------------------
def analyze_audio_emotion(waveform: np.ndarray, sampling_rate: int, models: Dict):
    fe = models["audio_feature_extractor"]
    model = models["audio_model"]

    target_sr = fe.sampling_rate
    if sampling_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=target_sr)

    waveform = force_min_length(waveform, target_length=target_sr)

    inputs = fe(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

    labels = [model.config.id2label[i] for i in range(len(probs))]
    return [{"label": labels[i], "score": float(probs[i])} for i in range(len(probs))]

def analyze_text_emotion(text: str, models: Dict):
    tok = models["text_tokenizer"]
    model = models["text_model"]

    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

    labels = [model.config.id2label[i] for i in range(len(probs))]
    return [{"label": labels[i], "score": float(probs[i])} for i in range(len(probs))]

def transcribe_with_whisper(models: Dict, waveform: np.ndarray, sr: int):
    whisper = models["whisper_model"]

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    waveform = waveform.astype(np.float32)
    try:
        segments, _ = whisper.transcribe(waveform, beam_size=5, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        st.error(f"Whisper エラー: {e}")
        return ""

# --------------------------
# Spotify 表示
# --------------------------
def search_spotify_and_render(emotion_label: str, sp):
    query = EMOTION_KEYWORD_MAP.get(emotion_label, emotion_label)
    st.subheader(f"🎧 プレイリスト（検索: {query}）")

    try:
        results = sp.search(q=f"{query} playlist", type="playlist", limit=6, market="JP")
    except Exception as e:
        st.error(f"Spotify 検索エラー: {e}")
        return

    for pl in results.get("playlists", {}).get("items", []):
        if not pl:
            continue
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                if pl.get("images"):
                    st.image(pl["images"][0]["url"])
            with cols[1]:
                name = pl.get("name")
                url = pl.get("external_urls", {}).get("spotify")
                st.markdown(f"**[{name}]({url})**")

# --------------------------
# WebRTC Audio Processor
# --------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        audio = frame.to_ndarray().astype(np.float32) / 32767.0
        mono = audio.mean(axis=1)  # ステレオ → モノラル
        self.frames.append(mono)
        return frame

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.title("🎙️ 音声感情 ＋ Whisper ＋ Spotify")

    models = load_models()
    sp = get_spotify_client()

    st.info("下のマイクウィンドウで録音し、停止後に自動解析されます。")

    webrtc_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if not webrtc_ctx.state.playing:
        st.stop()

    processor = webrtc_ctx.audio_processor
    if processor is None:
        st.stop()

    if st.button("録音を確定して解析する"):
        if not processor.frames:
            st.warning("音声がありません。話してから押してください。")
            st.stop()

        audio = np.concatenate(processor.frames).astype(np.float32)
        processor.frames = []

        wav_bytes = to_wav_bytes(audio)
        st.audio(wav_bytes, format="audio/wav")

        # Whisper
        with st.spinner("Whisper による文字起こし中..."):
            text = transcribe_with_whisper(models, audio, SAMPLE_RATE)
            st.write("🔤 文字起こし:", text)

        # 音声感情
        with st.spinner("音声感情推定中..."):
            ae = analyze_audio_emotion(audio, SAMPLE_RATE, models)
            top_a = max(ae, key=lambda x: x["score"])
            st.write("🔊 音声感情:", top_a)

        # テキスト感情
        if text:
            with st.spinner("テキスト感情推定中..."):
                te = analyze_text_emotion(text, models)
                top_t = max(te, key=lambda x: x["score"])
                st.write("📝 テキスト感情:", top_t)

        # Spotify
        chosen = None
        if text:
            chosen = top_t["label"]
        else:
            chosen = top_a["label"]

        if sp:
            search_spotify_and_render(chosen, sp)
        else:
            st.warning("Spotify 認証情報が未設定です。")

if __name__ == "__main__":
    main()
