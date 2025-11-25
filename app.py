import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import torch
import torchaudio
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
import os
import queue

# ==========================
# 定数と設定
# ==========================
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID", "ff259b9ec7f3420381662c278fed342f")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET", "a35403dc7fb64531ba6a98c5794fcef8")

AUDIO_EMOTION_MODEL_NAME = "superb/hubert-base-superb-er"
TEXT_EMOTION_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
WHISPER_MODEL_NAME = "base"
RECORDING_SAMPLING_RATE = 16000
MIN_AUDIO_LEN = 16000  # 1秒の波形

# ==========================
# ユーティリティ
# ==========================
def force_min_length(waveform, target_length=MIN_AUDIO_LEN):
    """波形を最低長にパディングして Hubert を壊さないようにする"""
    length = len(waveform)
    if length == 0:
        return np.zeros(target_length, dtype=np.float32)
    if length < target_length:
        pad = target_length - length
        return np.concatenate([waveform, np.zeros(pad, dtype=np.float32)])
    return waveform

# ==========================
# AIモデルのロード
# ==========================
@st.cache_resource
def load_models():
    with st.spinner("AIモデルをロードしています..."):
        feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_EMOTION_MODEL_NAME, trust_remote_code=True)
        audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_EMOTION_MODEL_NAME, trust_remote_code=True)
        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_EMOTION_MODEL_NAME)
        text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_EMOTION_MODEL_NAME)
        whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return {
        "audio_feature_extractor": feature_extractor,
        "audio_model": audio_model,
        "text_tokenizer": text_tokenizer,
        "text_model": text_model,
        "whisper_model": whisper_model,
    }

# ==========================
# 分析バックエンド
# ==========================
def analyze_audio_emotion(waveform, sampling_rate, models):
    feature_extractor = models["audio_feature_extractor"]
    model = models["audio_model"]

    # 必ず numpy → list にして渡す（Hubert 仕様）
    if sampling_rate != feature_extractor.sampling_rate:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=feature_extractor.sampling_rate)

    waveform = force_min_length(waveform)  # ★ 短すぎる音声を救済

    inputs = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    # ★ Hubert は [batch, seq] 形式でOK → unsqueeze(1) は絶対に要らない
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.nn.functional.softmax(logits, dim=-1)

    return [
        {"label": model.config.id2label[i], "score": scores[0][i].item()}
        for i in range(scores.shape[1])
    ]

def analyze_text_emotion(text, models):
    tokenizer = models["text_tokenizer"]
    model = models["text_model"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = torch.nn.functional.softmax(logits, dim=-1)
    return [
        {"label": model.config.id2label[i], "score": scores[0][i].item()}
        for i in range(scores.shape[1])
    ]

def transcribe_audio(waveform, models, lang="ja"):
    st.write(f"transcribe_audio called with waveform shape: {waveform.shape}, lang: {lang}")
    model = models["whisper_model"]
    segments, info = model.transcribe(waveform, beam_size=5, language=lang)
    transcribed_text = "".join([segment.text for segment in segments])
    st.write(f"Whisper transcription info: {info}")
    st.write(f"Whisper raw segments: {list(segments)}") # segmentsはイテレータなのでリストに変換
    st.write(f"Transcribed text (inside function): '{transcribed_text}'")
    return transcribed_text

# ==========================
# Spotify連携
# ==========================
@st.cache_resource
def get_spotify_client():
    try:
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
    except Exception as e:
        st.error(f"Spotifyへの接続に失敗しました: {e}")
        return None

EMOTION_KEYWORD_MAP = {
    "ang": "怒り, angry",
    "hap": "楽しい, happy",
    "neu": "落ち着く, neutral",
    "sad": "悲しい, sad",
    "joy": "喜び, joy",
    "optimism": "ポジティブ, positive",
    "anger": "怒り, angry",
    "sadness": "悲しみ, sad",
}

def search_spotify(emotion_label, sp):
    query = EMOTION_KEYWORD_MAP.get(emotion_label, emotion_label)
    st.subheader(f"🎧 「{emotion_label}」({query}) に関連するプレイリスト")
    results = sp.search(q=f"{query} プレイリスト", type="playlist", limit=5, market="JP")

    if not results["playlists"]["items"]:
        st.write("見つかりませんでした")
        return

    for playlist in results["playlists"]["items"]:
        col1, col2 = st.columns([1, 4])
        with col1:
            if playlist["images"]:
                st.image(playlist["images"][0]["url"])
        with col2:
            st.markdown(f"**[{playlist['name']}]({playlist['external_urls']['spotify']})** by {playlist['owner'].get('display_name', '不明')}")

# ==========================
# UI
# ==========================
st.set_page_config(layout="wide")
st.title("🎵 AI感情分析 ＆ Spotifyプレイリスト検索")

models = load_models()
sp = get_spotify_client()
if not sp:
    st.stop()

input_mode = st.radio("音声入力方法を選んでください：", ["🎙️ マイクで話す", "📁 音声ファイルをアップロード"], horizontal=True)

# --- 結果表示エリア ---
st.markdown("---")
st.header("分析結果")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🗣️ 文字起こし結果")
    text_display = st.empty()
with col2:
    st.subheader("🔊 音声の感情")
    audio_emotion_display = st.empty()
with col3:
    st.subheader("📝 テキストの感情")
    text_emotion_display = st.empty()

st.markdown("---")
playlist_display = st.container()

# ==========================
# 録音 or アップロード
# ==========================
if input_mode == "🎙️ マイクで話す":
    st.info("1. START → 2. 話す → 3. STOP で分析")

    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        media_stream_constraints={"video": False, "audio": True},
    )

    if not webrtc_ctx.state.playing:
        if "audio_buffer" in st.session_state and len(st.session_state.audio_buffer) > 0:
            with st.spinner("音声を分析しています..."):
                final_waveform = np.concatenate(st.session_state.audio_buffer)

                # ★ ここに追加（録音が短すぎる場合の防止）
                final_waveform = force_min_length(final_waveform, target_length=48000)  # 1秒ぶんを保証

                # リサンプリング
                resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=RECORDING_SAMPLING_RATE)
                final_waveform_16k = resampler(torch.from_numpy(final_waveform).float()).numpy()

                final_waveform_16k = force_min_length(final_waveform_16k)

                # 分析
                text = transcribe_audio(final_waveform_16k, models)
                audio_emotion = analyze_audio_emotion(final_waveform_16k, RECORDING_SAMPLING_RATE, models)
                text_emotion = analyze_text_emotion(text, models) if text else None

                # 表示
                text_display.write(text or "（なし）")
                if audio_emotion:
                    top = max(audio_emotion, key=lambda x: x["score"])
                    audio_emotion_display.success(f"**{top['label']}** ({top['score']:.2f})")
                if text_emotion:
                    top = max(text_emotion, key=lambda x: x["score"])
                    text_emotion_display.success(f"**{top['label']}** ({top['score']:.2f})")

                primary = (
                    max(audio_emotion, key=lambda x: x["score"])["label"]
                    if audio_emotion else (
                        max(text_emotion, key=lambda x: x["score"])["label"]
                        if text_emotion else None
                    )
                )
                if primary:
                    with playlist_display:
                        search_spotify(primary, sp)

            st.session_state.audio_buffer = []  # クリア
    st.info("🎙️ 録音中... STOPで分析開始")
    
    # ログ出力
    st.write("--- Debug Info (マイク入力) ---")
    st.write(f"webrtc_ctx.state.playing: {webrtc_ctx.state.playing}")

    if not webrtc_ctx.state.playing:
        if "audio_buffer" in st.session_state and len(st.session_state.audio_buffer) > 0:
            with st.spinner("音声を分析しています..."):
                final_waveform = np.concatenate(st.session_state.audio_buffer)
                st.write(f"Raw waveform shape: {final_waveform.shape}, dtype: {final_waveform.dtype}")

                # リサンプリング
                resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=RECORDING_SAMPLING_RATE)
                final_waveform_16k = resampler(torch.from_numpy(final_waveform).float()).numpy()
                st.write(f"Resampled waveform shape: {final_waveform_16k.shape}, dtype: {final_waveform_16k.dtype}")

                final_waveform_16k = force_min_length(final_waveform_16k)
                st.write(f"After force_min_length waveform shape: {final_waveform_16k.shape}")

                # 分析
                st.write("Calling transcribe_audio...")
                text = transcribe_audio(final_waveform_16k, models)
                st.write(f"Transcription result: '{text}'")
                
                audio_emotion = analyze_audio_emotion(final_waveform_16k, RECORDING_SAMPLING_RATE, models)
                text_emotion = analyze_text_emotion(text, models) if text else None

                # 表示
                text_display.write(text or "（なし）")
                if audio_emotion:
                    top = max(audio_emotion, key=lambda x: x["score"])
                    audio_emotion_display.success(f"**{top['label']}** ({top['score']:.2f})")
                if text_emotion:
                    top = max(text_emotion, key=lambda x: x["score"])
                    text_emotion_display.success(f"**{top['label']}** ({top['score']:.2f})")

                primary = (
                    max(audio_emotion, key=lambda x: x["score"])["label"]
                    if audio_emotion else (
                        max(text_emotion, key=lambda x: x["score"])["label"]
                        if text_emotion else None
                    )
                )
                if primary:
                    with playlist_display:
                        search_spotify(primary, sp)

            st.session_state.audio_buffer = []  # クリア
    else:
        # 録音中
        if "audio_buffer" not in st.session_state:
            st.session_state.audio_buffer = []

        st.info("🎙️ 録音中... STOPで分析開始")

        try:
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            for frame in frames:
                arr = frame.to_ndarray()
                if arr.ndim == 2:  # (samples, channels)
                    mono = arr.mean(axis=1)
                else:
                    mono = arr
                st.session_state.audio_buffer.append(mono.astype(np.float32))
            st.write(f"Current audio_buffer size: {len(st.session_state.audio_buffer)}")
        except queue.Empty:
            st.write("Audio queue empty.")

else:
    uploaded_file = st.file_uploader("音声ファイル(mp3, wav) をアップロード", type=["wav", "mp3"])

    if uploaded_file:
        with st.spinner("音声を分析しています..."):
            waveform, _ = librosa.load(uploaded_file, sr=RECORDING_SAMPLING_RATE, mono=True)
            waveform = force_min_length(waveform)
            st.write("--- Debug Info (ファイルアップロード) ---")
            st.write(f"Uploaded waveform shape: {waveform.shape}, dtype: {waveform.dtype}")

            st.write("Calling transcribe_audio...")
            text = transcribe_audio(waveform, models)
            st.write(f"Transcription result: '{text}'")
