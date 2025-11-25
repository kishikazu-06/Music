import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import torch
import torchaudio
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
import av
import os
import queue
import time

# ==========================
# 定数と設定
# ==========================
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID", "ff259b9ec7f3420381662c278fed342f")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET", "a35403dc7fb64531ba6a98c5794fcef8")

AUDIO_EMOTION_MODEL_NAME = "superb/hubert-base-superb-er"
TEXT_EMOTION_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
WHISPER_MODEL_NAME = "base"
RECORDING_SAMPLING_RATE = 16000

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
    if sampling_rate != feature_extractor.sampling_rate:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=feature_extractor.sampling_rate)
    # Convert numpy array to 2D PyTorch tensor (batch_size, sequence_length)
    torch_waveform = torch.from_numpy(waveform).float()
    if torch_waveform.ndim == 1:
        torch_waveform = torch_waveform.unsqueeze(0)

    inputs = feature_extractor(torch_waveform, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.nn.functional.softmax(logits, dim=-1)
    predictions = [{"label": model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores[0])]
    return sorted(predictions, key=lambda x: x["score"], reverse=True)

def analyze_text_emotion(text, models):
    tokenizer = models["text_tokenizer"]
    model = models["text_model"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.nn.functional.softmax(logits, dim=-1)
    predictions = [{"label": model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores[0])]
    return sorted(predictions, key=lambda x: x["score"], reverse=True)

def transcribe_audio(waveform, models, lang="ja"):
    model = models["whisper_model"]
    segments, _ = model.transcribe(waveform, beam_size=5, language=lang)
    return "".join([segment.text for segment in segments])

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

EMOTION_KEYWORD_MAP = {"ang": "怒り, angry", "hap": "楽しい, happy", "neu": "落ち着く, neutral", "sad": "悲しい, sad", "joy": "喜び, joy", "optimism": "ポジティブ, positive", "anger": "怒り, angry", "sadness": "悲しみ, sad"}

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

input_mode = st.radio("音声入力方法を選んでください：", ["🎙️ マイクで話す", "📁 音声ファイルをアップロード"], horizontal=True, key="input_mode")

# --- 結果表示エリアを先に定義 ---
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

# --- 録音/アップロード処理 ---
if input_mode == "🎙️ マイクで話す":
    st.info("1. STARTを押す → 2. 話す → 3. STOPを押す")
    
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        media_stream_constraints={"video": False, "audio": True},
    )

    if not webrtc_ctx.state.playing:
        # STOPが押された後、または初期状態
        if "audio_buffer" in st.session_state and len(st.session_state.audio_buffer) > 0:
            with st.spinner("音声データを処理・分析しています..."):
                final_waveform = np.concatenate(st.session_state.audio_buffer)
                
                # リサンプリング
                resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=RECORDING_SAMPLING_RATE)
                final_waveform_16k = resampler(torch.from_numpy(final_waveform).float()).numpy()

                # 分析実行
                text = transcribe_audio(final_waveform_16k, models)
                audio_emotion = analyze_audio_emotion(final_waveform_16k, RECORDING_SAMPLING_RATE, models)
                text_emotion = analyze_text_emotion(text, models) if text else None

                # 結果表示
                text_display.write(text or "（なし）")
                if audio_emotion:
                    audio_emotion_display.success(f"**{audio_emotion[0]['label']}** ({audio_emotion[0]['score']:.2f})")
                if text_emotion:
                    text_emotion_display.success(f"**{text_emotion[0]['label']}** ({text_emotion[0]['score']:.2f})")
                
                primary_emotion_label = audio_emotion[0]['label'] if audio_emotion else (text_emotion[0]['label'] if text_emotion else None)
                if primary_emotion_label:
                    with playlist_display:
                         search_spotify(primary_emotion_label, sp)

            # 処理が終わったらバッファをクリア
            st.session_state.audio_buffer = [] 
    else:
        # 録音開始時にバッファを初期化
        if "audio_buffer" not in st.session_state or len(st.session_state.get("audio_buffer", [])) > 0:
            st.session_state.audio_buffer = []

        st.info("🎙️ 録音中... 停止すると分析が始まります。")
        
        # 音声フレームをバッファに溜める
        try:
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            for frame in frames:
                st.session_state.audio_buffer.append(frame.to_ndarray().mean(axis=1))
        except queue.Empty:
            pass

elif input_mode == "📁 音声ファイルをアップロード":
    uploaded_file = st.file_uploader("音声ファイル(mp3, wav) をアップロードしてください", type=["wav", "mp3"])

    if uploaded_file:
        with st.spinner("音声ファイルを処理・分析しています..."):
            waveform, _ = librosa.load(uploaded_file, sr=RECORDING_SAMPLING_RATE, mono=True)
            text = transcribe_audio(waveform, models)
            audio_emotion = analyze_audio_emotion(waveform, RECORDING_SAMPLING_RATE, models)
            text_emotion = analyze_text_emotion(text, models) if text else None
            
            # 結果表示
            text_display.write(text or "（なし）")
            if audio_emotion:
                audio_emotion_display.success(f"**{audio_emotion[0]['label']}** ({audio_emotion[0]['score']:.2f})")
            if text_emotion:
                text_emotion_display.success(f"**{text_emotion[0]['label']}** ({text_emotion[0]['score']:.2f})")
            
            primary_emotion_label = audio_emotion[0]['label'] if audio_emotion else (text_emotion[0]['label'] if text_emotion else None)
            if primary_emotion_label:
                with playlist_display:
                    search_spotify(primary_emotion_label, sp)
