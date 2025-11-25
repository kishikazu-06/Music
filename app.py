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
    st.session_state.update({"text": "", "audio_emotion": None, "text_emotion": None, "audio_frames": []})

if input_mode == "🎙️ マイクで話す":
    st.info("1. STARTを押す → 2. マイクを許可する → 3. 話す → 4. STOPを押す")
    
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()
    
    was_playing = st.session_state.get("is_playing", False)
    is_playing = webrtc_ctx.state.playing
    st.session_state["is_playing"] = is_playing

    if is_playing:
        status_indicator.info("🎙️ 録音中... 音声フレームを収集しています。")
        if "audio_frames" not in st.session_state:
            st.session_state.audio_frames = []
        
        # コールバックの代わりにフレームを直接取得
        if webrtc_ctx.audio_receiver:
            try:
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                for frame in frames:
                    st.session_state.audio_frames.append(frame.to_ndarray())
            except queue.Empty:
                pass # タイムアウトは無視

    if not is_playing and was_playing:
        status_indicator.info("録音停止。分析を開始します...")
        
        frames = st.session_state.get("audio_frames", [])
        if len(frames) > 0:
            with st.spinner("音声データを処理・分析しています..."):
                sound_chunks = [frame.mean(axis=1) for frame in frames] # モノラル化
                final_waveform = np.concatenate(sound_chunks)
                
                resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=RECORDING_SAMPLING_RATE)
                final_waveform_16k = resampler(torch.from_numpy(final_waveform).float()).numpy()
                
                st.session_state.text = transcribe_audio(final_waveform_16k, models)
                st.session_state.audio_emotion = analyze_audio_emotion(final_waveform_16k, RECORDING_SAMPLING_RATE, models)
                if st.session_state.text:
                    st.session_state.text_emotion = analyze_text_emotion(st.session_state.text, models)
            
            st.session_state.audio_frames = []
            st.rerun() # 結果を表示するために再実行
        else:
            status_indicator.warning("音声が録音されなかったようです。")

elif input_mode == "📁 音声ファイルをアップロード":
    uploaded_file = st.file_uploader("音声ファイル(mp3, wav) をアップロードしてください", type=["wav", "mp3"])

    if uploaded_file:
        with st.spinner("音声ファイルを処理・分析しています..."):
            try:
                waveform, _ = librosa.load(uploaded_file, sr=RECORDING_SAMPLING_RATE, mono=True)
                st.session_state.text = transcribe_audio(waveform, models)
                st.session_state.audio_emotion = analyze_audio_emotion(waveform, RECORDING_SAMPLING_RATE, models)
                if st.session_state.text:
                    st.session_state.text_emotion = analyze_text_emotion(st.session_state.text, models)
            except Exception as e:
                st.error(f"ファイル処理中にエラーが発生しました: {e}")
        st.rerun()

# --- 結果表示 ---
if st.session_state.get("text") or st.session_state.get("audio_emotion"):
    st.markdown("---")
    st.header("分析結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🗣️ 文字起こし結果")
        st.write(st.session_state.text or "（なし）")
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

    # 結果を表示したらクリアする
    st.session_state.text = ""
    st.session_state.audio_emotion = None
    st.session_state.text_emotion = None
