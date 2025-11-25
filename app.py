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
import tempfile
import os
import soundfile as sf
from scipy.io.wavfile import write as write_wav

# ==========================
# 定数と設定
# ==========================
# Spotify認証情報（環境変数からの読み込みを推奨）
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID", "ff259b9ec7f3420381662c278fed342f")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET", "a35403dc7fb64531ba6a98c5794fcef8")

# AIモデルの設定
AUDIO_EMOTION_MODEL_NAME = "superb/hubert-base-superb-er"
TEXT_EMOTION_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
WHISPER_MODEL_NAME = "base" # "base", "small", "medium", "large" から選択

# 音声録音の設定
RECORDING_SAMPLING_RATE = 16000
RECORDING_TIMEOUT_SECONDS = 10 # 10秒間音声が来なかったらタイムアウト

# ==========================
# AIモデルのロード（キャッシュ利用）
# ==========================
@st.cache_resource
def load_models():
    """AIモデルをロードし、キャッシュする"""
    # 音声感情分析モデル
    feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_EMOTION_MODEL_NAME, trust_remote_code=True)
    audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_EMOTION_MODEL_NAME, trust_remote_code=True)
    
    # テキスト感情分析モデル
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_EMOTION_MODEL_NAME)
    text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_EMOTION_MODEL_NAME)

    # 音声認識モデル
    whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    
    return {
        "audio_feature_extractor": feature_extractor,
        "audio_model": audio_model,
        "text_tokenizer": text_tokenizer,
        "text_model": text_model,
        "whisper_model": whisper_model,
    }

# ==========================
# 感情分析バックエンド
# ==========================
def analyze_audio_emotion(waveform, sampling_rate, models):
    """音声波形データから感情を分析する"""
    feature_extractor = models["audio_feature_extractor"]
    model = models["audio_model"]

    # モデルの期待するサンプリングレートに変換
    if sampling_rate != feature_extractor.sampling_rate:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=feature_extractor.sampling_rate)

    inputs = feature_extractor(waveform, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    scores = torch.nn.functional.softmax(logits, dim=-1)
    predictions = [{"label": model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores[0])]
    return sorted(predictions, key=lambda x: x["score"], reverse=True)

def analyze_text_emotion(text, models):
    """テキストから感情を分析する"""
    tokenizer = models["text_tokenizer"]
    model = models["text_model"]
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    scores = torch.nn.functional.softmax(logits, dim=-1)
    predictions = [{"label": model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores[0])]
    return sorted(predictions, key=lambda x: x["score"], reverse=True)

def transcribe_audio(waveform, models):
    """音声波形データをテキストに変換する"""
    model = models["whisper_model"]
    segments, _ = model.transcribe(waveform, beam_size=5, language="ja")
    return "".join([segment.text for segment in segments])

# ==========================
# Spotify連携
# ==========================
@st.cache_resource
def get_spotify_client():
    """Spotifyクライアントを取得"""
    try:
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        ))
    except Exception as e:
        st.error(f"Spotifyへの接続に失敗しました。認証情報を確認してください。: {e}")
        return None

# 感情とキーワードのマッピング
EMOTION_KEYWORD_MAP = {
    "ang": "怒り, angry, 激しい",
    "hap": "楽しい, happy, 明るい",
    "neu": "落ち着く, neutral, 静か",
    "sad": "悲しい, sad, 切ない",
    "joy": "喜び, joy, ワクワク",
    "optimism": "ポジティブ, positive, 元気",
    "anger": "怒り, angry, 激しい",
    "sadness": "悲しみ, sad, 切ない",
}

def search_spotify(emotion_label, sp):
    """感情ラベルに基づいてSpotifyでプレイリストを検索"""
    query = EMOTION_KEYWORD_MAP.get(emotion_label, emotion_label)
    st.subheader(f"🎧 「{emotion_label}」({query}) に関連するプレイリスト")
    
    results = sp.search(q=f"{query} プレイリスト", type="playlist", limit=5, market="JP")
    playlists = results["playlists"]["items"]

    if not playlists:
        st.write("見つかりませんでした")
        return

    for playlist in playlists:
        col1, col2 = st.columns([1, 4])
        with col1:
            if playlist["images"]:
                st.image(playlist["images"][0]["url"])
        with col2:
            st.markdown(f"**[{playlist['name']}]({playlist['external_urls']['spotify']})**")
            st.write(f"by {playlist['owner'].get('display_name', '不明')}")


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(layout="wide")
st.title("🎵 AI感情分析 ＆ Spotifyプレイリスト検索")
st.write("マイクや音声ファイルからAIが感情を分析し、あなたに合った音楽を提案します。")

# --- モデルのロード ---
with st.spinner("AIモデルをロードしています..."):
    models = load_models()
st.success("AIモデルの準備ができました！")

# --- Spotifyクライアントの準備 ---
sp = get_spotify_client()
if sp is None:
    st.stop()

# --- 入力方法の選択 ---
input_mode = st.radio("音声入力方法を選んでください：", ["🎙️ マイクで話す", "📁 音声ファイルをアップロード"], horizontal=True)

# --- セッションステートの初期化 ---
if "text" not in st.session_state:
    st.session_state["text"] = ""
if "audio_emotion" not in st.session_state:
    st.session_state["audio_emotion"] = None
if "text_emotion" not in st.session_state:
    st.session_state["text_emotion"] = None

# ==========================
# 🎤 マイク入力モード
# ==========================
if input_mode == "🎙️ マイクで話す":
    st.info("下の『START』ボタンを押してマイクに向かって話してください。話をやめると自動で分析が始まります。")

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        st.session_state["text"] = ""
        st.session_state["audio_emotion"] = None
        st.session_state["text_emotion"] = None

    if webrtc_ctx.audio_receiver:
        status_indicator.info("マイクで話してください...")
        
        sound_chunk = st.empty()
        
        try:
            audio_frames = []
            last_received_time = time.time()

            while True:
                try:
                    frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                    audio_frames.append(frame)
                    last_received_time = time.time()
                except av.error.TimeoutError:
                    current_time = time.time()
                    if current_time - last_received_time > RECORDING_TIMEOUT_SECONDS:
                        status_indicator.info("音声が途絶えたため、分析を開始します。")
                        break
            
            if audio_frames:
                status_indicator.info("音声データを処理・分析しています...")

                # フレームを結合してNumpy配列に変換
                sound = np.concatenate([f.to_ndarray() for f in audio_frames])
                sound = sound.mean(axis=1) # ステレオをモノラルに
                
                # サンプリングレートをWhisper用に変換
                resampler = torchaudio.transforms.Resample(
                    orig_freq=webrtc_ctx.audio_receiver.format.sample_rate, 
                    new_freq=RECORDING_SAMPLING_RATE
                )
                waveform_16k = resampler(torch.from_numpy(sound).float()).numpy()

                # --- AI分析実行 ---
                st.session_state["text"] = transcribe_audio(waveform_16k, models)
                st.session_state["audio_emotion"] = analyze_audio_emotion(waveform_16k, RECORDING_SAMPLING_RATE, models)
                if st.session_state["text"]:
                    st.session_state["text_emotion"] = analyze_text_emotion(st.session_state["text"], models)
                
                status_indicator.success("分析が完了しました！")
                # 再実行して結果を表示
                st.experimental_rerun()

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# ==========================
# 📁 アップロード音声モード
# ==========================
elif input_mode == "📁 音声ファイルをアップロード":
    uploaded_file = st.file_uploader("音声ファイル(mp3, wav) をアップロードしてください", type=["wav", "mp3"])

    if uploaded_file:
        with st.spinner("音声ファイルを処理・分析しています..."):
            try:
                # 一時ファイルに保存して処理
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                waveform, sampling_rate = librosa.load(audio_path, sr=RECORDING_SAMPLING_RATE, mono=True)
                
                # --- AI分析実行 ---
                st.session_state["text"] = transcribe_audio(waveform, models)
                st.session_state["audio_emotion"] = analyze_audio_emotion(waveform, sampling_rate, models)
                if st.session_state["text"]:
                    st.session_state["text_emotion"] = analyze_text_emotion(st.session_state["text"], models)

            except Exception as e:
                st.error(f"ファイル処理中にエラーが発生しました: {e}")
            finally:
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)

# ==========================
# 分析結果の表示 & Spotify検索
# ==========================
if st.session_state.get("text") or st.session_state.get("audio_emotion"):
    st.markdown("---")
    st.header("分析結果")

    col1, col2, col3 = st.columns(3)

    # --- 1. 音声認識の結果 ---
    with col1:
        st.subheader("🗣️ 音声認識 (Whisper)")
        st.write(st.session_state.get("text", "（認識結果がありません）"))

    # --- 2. 音声感情分析の結果 ---
    with col2:
        st.subheader("🔊 音声の感情")
        audio_emotion = st.session_state.get("audio_emotion")
        if audio_emotion:
            primary_emotion = audio_emotion[0]
            st.success(f"**{primary_emotion['label']}** ({primary_emotion['score']:.2f})")
            st.write("その他:")
            for emo in audio_emotion[1:]:
                st.text(f"{emo['label']}: {emo['score']:.2f}")
        else:
            st.write("（分析結果がありません）")

    # --- 3. テキスト感情分析の結果 ---
    with col3:
        st.subheader("📝 テキストの感情")
        text_emotion = st.session_state.get("text_emotion")
        if text_emotion:
            primary_emotion = text_emotion[0]
            st.success(f"**{primary_emotion['label']}** ({primary_emotion['score']:.2f})")
            st.write("その他:")
            for emo in text_emotion[1:]:
                st.text(f"{emo['label']}: {emo['score']:.2f}")
        else:
            st.write("（分析結果がありません）")
    
    st.markdown("---")

    # --- Spotify検索 ---
    # 音声感情を優先し、なければテキスト感情を使う
    primary_emotion_label = None
    if st.session_state.get("audio_emotion"):
        primary_emotion_label = st.session_state["audio_emotion"][0]["label"]
    elif st.session_state.get("text_emotion"):
        primary_emotion_label = st.session_state["text_emotion"][0]["label"]

    if primary_emotion_label:
        search_spotify(primary_emotion_label, sp)
