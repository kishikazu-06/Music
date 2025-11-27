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
#
