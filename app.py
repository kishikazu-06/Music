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
import time
import queue
import tempfile
import os

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
    inputs = feature_extractor(waveform, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
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

input_mode = st.radio("音声入力方法を選んでください：", ["🎙️ マイクで話す", "📁 音声ファイルをアップロード"], horizontal=True)

if "text" not in st.session_state:
    st.session_state.update({"text": "", "audio_emotion": None, "text_emotion": None})

if input_mode == "🎙️ マイクで話す":
    st.info("下の『START』を押してマイクに向かって話してください。")
    
    if "audio_frames_queue" not in st.session_state:
        st.session_state.audio_frames_queue = queue.Queue()

    def audio_frame_callback(frame: av.AudioFrame):
        print(f"★DEBUG 1: Audio frame received: {frame.format.name} {frame.layout.name} {frame.samples}")
        st.session_state.audio_frames_queue.put(frame.to_ndarray())

    webrtc_ctx = webrtc_streamer(key="speech-to-text-realtime", mode=WebRtcMode.SENDONLY, audio_frame_callback=audio_frame_callback, media_stream_constraints={"video": False, "audio": True})

    status_indicator = st.empty()
    realtime_text_display = st.empty()

    if webrtc_ctx.state.playing:
        status_indicator.info("🎙️ 録音中...")
        if "audio_buffer" not in st.session_state:
            st.session_state.audio_buffer = np.array([], dtype=np.float32)
        if "full_audio" not in st.session_state:
            st.session_state.full_audio = np.array([], dtype=np.float32)

        while webrtc_ctx.state.playing:
            try:
                frame_data = st.session_state.audio_frames_queue.get(timeout=1.0)
                print(f"★DEBUG 2: Got frame from queue. Shape: {frame_data.shape}, Dtype: {frame_data.dtype}")
                sound_chunk = frame_data.mean(axis=1)
                st.session_state.audio_buffer = np.append(st.session_state.audio_buffer, sound_chunk)
                st.session_state.full_audio = np.append(st.session_state.full_audio, sound_chunk)

                buffer_len = len(st.session_state.audio_buffer)
                threshold = 48000 * 1.5
                print(f"★DEBUG 3: Current buffer length: {buffer_len}, Threshold: {threshold}")

                if buffer_len > threshold:
                    print("★DEBUG 4: --- Triggering transcription ---")
                    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=RECORDING_SAMPLING_RATE)
                    waveform_16k = resampler(torch.from_numpy(st.session_state.audio_buffer).float()).numpy()
                    text = transcribe_audio(waveform_16k, models)
                    
                    if "realtime_text" not in st.session_state:
                        st.session_state.realtime_text = ""
                    st.session_state.realtime_text += text
                    realtime_text_display.markdown(f"**リアルタイム:** {st.session_state.realtime_text}")
                    st.session_state.audio_buffer = np.array([], dtype=np.float32)
            except queue.Empty:
                continue

        status_indicator.info("録音停止。最終分析を実行しています...")
        if len(st.session_state.full_audio) > 0:
            resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=RECORDING_SAMPLING_RATE)
            final_waveform_16k = resampler(torch.from_numpy(st.session_state.full_audio).float()).numpy()

            st.session_state.text = transcribe_audio(final_waveform_16k, models)
            st.session_state.audio_emotion = analyze_audio_emotion(final_waveform_16k, RECORDING_SAMPLING_RATE, models)
            if st.session_state.text:
                st.session_state.text_emotion = analyze_text_emotion(st.session_state.text, models)

        for key in ["audio_buffer", "full_audio", "realtime_text", "audio_frames_queue"]:
            if key in st.session_state:
                del st.session_state[key]
        
        time.sleep(0.5)
        st.rerun()
    else:
        status_indicator.info("▶️ 『START』を押して録音を開始してください。")

elif input_mode == "📁 音声ファイルをアップロード":
    uploaded_file = st.file_uploader("音声ファイル(mp3, wav) をアップロードしてください", type=["wav", "mp3"])

    if uploaded_file:
        with st.spinner("音声ファイルを処理・分析しています..."):
            try:
                waveform, sampling_rate = librosa.load(uploaded_file, sr=RECORDING_SAMPLING_RATE, mono=True)
                st.session_state.text = transcribe_audio(waveform, models)
                st.session_state.audio_emotion = analyze_audio_emotion(waveform, sampling_rate, models)
                if st.session_state.text:
                    st.session_state.text_emotion = analyze_text_emotion(st.session_state.text, models)
            except Exception as e:
                st.error(f"ファイル処理中にエラーが発生しました: {e}")

if st.session_state.get("text") or st.session_state.get("audio_emotion"):
    st.markdown("---")
    st.header("分析結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🗣️ 文字起こし結果")
        st.write(st.session_state.text)
    with col2:
        st.subheader("🔊 音声の感情")
        if st.session_state.audio_emotion:
            primary_emotion = st.session_state.audio_emotion[0]
            st.success(f"**{primary_emotion['label']}** ({primary_emotion['score']:.2f})")
    with col3:
        st.subheader("📝 テキストの感情")
        if st.session_state.text_emotion:
            primary_emotion = st.session_state.text_emotion[0]
            st.success(f"**{primary_emotion['label']}** ({primary_emotion['score']:.2f})")
    
    st.markdown("---")
    primary_emotion_label = None
    if st.session_state.get("audio_emotion"):
        primary_emotion_label = st.session_state["audio_emotion"][0]["label"]
    elif st.session_state.get("text_emotion"):
        primary_emotion_label = st.session_state["text_emotion"][0]["label"]
    if primary_emotion_label and sp:
        search_spotify(primary_emotion_label, sp)