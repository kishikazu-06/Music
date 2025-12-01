# app.py
import os
import io
import threading
import wave
from typing import Optional, List, Dict

import numpy as np
import streamlit as st
import sounddevice as sd
import librosa
import torch
import torchaudio
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
SAMPLE_RATE = 16000  # アプリ内で扱う基本サンプルレート（Whisper 用）
MIN_AUDIO_LEN = SAMPLE_RATE  # 最低 1 秒分を保証
WHISPER_SIZE = "tiny"  # 必要なら small/medium/large-v3 等に変更
SPOTIPY_CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")

# マッピング（感情 -> 検索クエリの例）
EMOTION_KEYWORD_MAP = {
    "anger": "angry 怒り",
    "sadness": "sad 悲しい",
    "joy": "happy 楽しい",
    "optimism": "positive ポジティブ",
    "neutral": "calm neutral",
    "surprise": "surprise",
    "love": "love",
    # モデルによってラベル名が異なる場合があるので適宜調整
}

from pydub import AudioSegment


# --------------------------
# ユーティリティ関数
# --------------------------
def process_uploaded_audio(uploaded_file) -> Optional[np.ndarray]:
    """UploadedFile オブジェクトを読み込み、音声波形(numpy)に変換する"""
    try:
        audio = AudioSegment.from_file(uploaded_file)
        # モノラルに変換
        audio = audio.set_channels(1)
        # サンプルレートを変換
        audio = audio.set_frame_rate(SAMPLE_RATE)
        
        # NumPy 配列に変換
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / np.iinfo(audio.sample_width).max
        return samples
    except Exception as e:
        st.error(f"音声ファイルの処理中にエラーが発生しました: {e}")
        return None

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
# モデル読み込み（キャッシュ）
# --------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Audio emotion (Hubert), Text emotion (Roberta-twitter), Whisper"""
    with st.spinner("モデルをロードしています（時間がかかります）..."):
        # 音声感情モデル
        audio_fe = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-er", trust_remote_code=True)
        audio_model = AutoModelForAudioClassification.from_pretrained("superb/hubert-base-superb-er", trust_remote_code=True)

        # テキスト感情モデル
        text_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        text_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

        # Whisper
        whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    return {
        "audio_feature_extractor": audio_fe,
        "audio_model": audio_model,
        "text_tokenizer": text_tok,
        "text_model": text_model,
        "whisper_model": whisper_model,
    }


# --------------------------
# Spotipy client
# --------------------------
@st.cache_resource
def get_spotify_client():
    if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
        return None
    try:
        auth = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
        return spotipy.Spotify(auth_manager=auth)
    except Exception as e:
        st.error(f"Spotify 認証に失敗しました: {e}")
        return None


# --------------------------
# 推論関数
# --------------------------
def analyze_audio_emotion(waveform: np.ndarray, sampling_rate: int, models: Dict) -> List[Dict]:
    """waveform: 1D numpy float32, sampling_rate: waveform の SR"""
    fe = models["audio_feature_extractor"]
    model = models["audio_model"]

    # リサンプリングが必要なら行う
    target_sr = fe.sampling_rate if hasattr(fe, "sampling_rate") else 16000
    if sampling_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=target_sr)

    waveform = force_min_length(waveform, target_length=target_sr)  # 最低長保証

    # feature extractor に渡す
    inputs = fe(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

    labels = [model.config.id2label[i] for i in range(len(probs))]
    return [{"label": labels[i], "score": float(probs[i])} for i in range(len(probs))]


def analyze_text_emotion(text: str, models: Dict) -> List[Dict]:
    tok = models["text_tokenizer"]
    model = models["text_model"]
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    labels = [model.config.id2label[i] for i in range(len(probs))]
    return [{"label": labels[i], "score": float(probs[i])} for i in range(len(probs))]


def transcribe_with_whisper(models: Dict, waveform: np.ndarray, sr: int, language: Optional[str] = None) -> str:
    """waveform must be 1D float32. We ensure it is at SAMPLE_RATE before passing to whisper."""
    whisper = models["whisper_model"]

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    waveform = waveform.astype(np.float32)
    try:
        segments, _ = whisper.transcribe(waveform, language=language, beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    except Exception as e:
        st.error(f"Whisper エラー: {e}")
        return ""


# --------------------------
# Spotify 検索表示
# --------------------------
def search_spotify_and_render(emotion_label: str, sp: spotipy.Spotify):
    if sp is None:
        st.warning("Spotify クライアントが設定されていません。環境変数 SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET を設定してください。")
        return

    query = EMOTION_KEYWORD_MAP.get(emotion_label, emotion_label)
    st.subheader(f"🎧 「{emotion_label}」に合うプレイリスト候補（検索: {query}）")

    try:
        results = sp.search(q=f"{query} playlist", type="playlist", limit=6, market="JP")
    except Exception as e:
        st.error(f"Spotify 検索に失敗しました: {e}")
        return

    playlists = results.get("playlists", {}).get("items", [])
    if not playlists:
        st.write("プレイリストが見つかりませんでした。")
        return

    for pl in playlists:
        # None 安全対策
        if not pl:
            continue

        cols = st.columns([1, 4])
        with cols[0]:
            images = pl.get("images") if isinstance(pl, dict) else None
            if images:
                st.image(images[0]["url"], use_container_width=True)
            else:
                st.write("(画像なし)")

        with cols[1]:
            name = pl.get("name", "名称不明")
            url = pl.get("external_urls", {}).get("spotify", "#")
            owner = pl.get("owner", {}).get("display_name", "不明")
            st.markdown(f"**[{name}]({url})** — {owner}")



# --------------------------
# 録音スレッド（既に動作するものを流用）
# --------------------------
def record_audio_thread(audio_buffer: list, stop_event: threading.Event):
    def callback(indata, frames, time, status):
        if status:
            print("InputStream status:", status)
        if not stop_event.is_set():
            # indata is shape (frames, channels)
            audio_buffer.append(indata.copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
            while not stop_event.is_set():
                sd.sleep(50)
    except Exception as e:
        print(f"録音スレッドエラー: {e}")
        # エラーは UI 側で確認できるように print に出す


def display_analysis_results(waveform: np.ndarray, sr: int, models: Dict, sp: spotipy.Spotify):
    """音声波形を受け取り、分析〜結果表示までの一連の流れを実行する"""
    st.success("音声を取得しました！")
    
    # 再生とダウンロード用に WAV データを準備
    wav_bytes = to_wav_bytes(waveform, sr)
    st.audio(wav_bytes, format="audio/wav")

    # --- Whisper に渡す前の保険: 最低長確保 ---
    final_waveform = force_min_length(waveform, target_length=MIN_AUDIO_LEN)
    
    last_text = ""
    last_audio_emotion = None
    last_text_emotion = None

    # --- 文字起こし ---
    with st.spinner("Whisper で文字起こし中..."):
        text = transcribe_with_whisper(models, final_waveform, sr, language=None)
        last_text = text
        st.write("🔤 文字起こし結果:")
        st.write(text or "（テキスト無し）")

    # --- 音声感情推定（Hubert） ---
    with st.spinner("音声から感情推定中..."):
        audio_emotion = analyze_audio_emotion(final_waveform, sr, models)
        last_audio_emotion = audio_emotion
        if audio_emotion:
            audio_top = max(audio_emotion, key=lambda x: x["score"])
            st.write(f"🔊 音声感情（トップ）: **{audio_top['label']}** ({audio_top['score']:.2f})")
            with st.expander("詳細を見る"):
                st.dataframe(audio_emotion)

    # --- テキスト感情推定（Roberta） ---
    if text:
        with st.spinner("テキストから感情推定中..."):
            text_emotion = analyze_text_emotion(text, models)
            last_text_emotion = text_emotion
            if text_emotion:
                text_top = max(text_emotion, key=lambda x: x["score"])
                st.write(f"📝 テキスト感情（トップ）: **{text_top['label']}** ({text_top['score']:.2f})")
                with st.expander("詳細を見る"):
                    st.dataframe(text_emotion)
    else:
        st.info("テキストが空なのでテキストベースの感情推定はスキップしました。")

    # --- Spotify で選曲 ---
    if sp is None:
        st.warning("Spotify クライアントが未設定です（環境変数を確認）。プレイリスト表示はできません。")
    else:
        chosen_label = None
        # 優先基準：音声感情のスコアが高ければ音声を優先
        if last_audio_emotion:
            audio_top_score = max(last_audio_emotion, key=lambda x: x["score"])["score"]
            if audio_top_score > 0.3: # ある程度の確信度がある場合
                 chosen_label = max(last_audio_emotion, key=lambda x: x["score"])["label"]
        
        # テキスト感情が非常に明確な場合はそちらを優先
        if last_text_emotion:
            text_top = max(last_text_emotion, key=lambda x: x["score"])
            if text_top["score"] > 0.7: # テキスト感情が非常に高い確信度を持つ場合
                chosen_label = text_top["label"]

        # どちらも確信度が低い場合、デフォルトで音声感情を選ぶ
        if chosen_label is None and last_audio_emotion:
            chosen_label = max(last_audio_emotion, key=lambda x: x["score"])["label"]

        if chosen_label:
            search_spotify_and_render(chosen_label, sp)
        else:
            st.info("感情の優位ラベルが特定できませんでした。")


# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(page_title="音声→感情→プレイリスト", layout="wide")
    st.title("🎙️ 音声感情推定 + Whisper テキスト感情 + Spotify 選曲")

    models = load_models()
    sp = get_spotify_client()

    # --- サイドバー ---
    with st.sidebar:
        st.header("設定")
        # モデル選択は現在固定だが、拡張性のためにUIは残す
        st.selectbox("Whisper model size", ["small", "medium"], index=0, disabled=True)
        st.info(f"現在 `WHISPER_SIZE = '{WHISPER_SIZE}'` に固定されています。")
        st.write("内部サンプルレート:", SAMPLE_RATE)

    # --- UI タブ ---
    tab1, tab2 = st.tabs(["🎤 マイクから録音", "⬆️ ファイルをアップロード"])

    # --- タブ1: マイク録音 ---
    with tab1:
        st.header("マイクを使ってリアルタイムで分析")
        
        # セッション state 初期化
        if "recording" not in st.session_state:
            st.session_state.recording = False
        if "audio_buffer" not in st.session_state:
            st.session_state.audio_buffer = []
        if "stop_event" not in st.session_state:
            st.session_state.stop_event = threading.Event()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 録音開始", disabled=st.session_state.recording, key="start_rec"):
                st.session_state.recording = True
                st.session_state.audio_buffer = []
                st.session_state.stop_event.clear()
                threading.Thread(target=record_audio_thread, args=(st.session_state.audio_buffer, st.session_state.stop_event), daemon=True).start()
                st.rerun()
        with col2:
            if st.button("⏹ 録音停止", disabled=not st.session_state.recording, key="stop_rec"):
                st.session_state.stop_event.set()
                st.session_state.recording = False
                st.rerun()

        if st.session_state.recording:
            st.info("録音中... 話が終わったら「録音停止」を押してください。")

        # 録音が完了して buffer がある場合は解析
        if not st.session_state.recording and st.session_state.audio_buffer:
            with st.spinner("録音データを処理中..."):
                frames = [chunk.squeeze() if chunk.ndim == 2 else chunk for chunk in st.session_state.audio_buffer]
                final_waveform = np.concatenate([f.reshape(-1) for f in frames]).astype(np.float32)
            
            # buffer をクリア
            st.session_state.audio_buffer = []
            
            # 分析と結果表示
            display_analysis_results(final_waveform, SAMPLE_RATE, models, sp)

    # --- タブ2: ファイルアップロード ---
    with tab2:
        st.header("音声ファイルをアップロードして分析")
        uploaded_file = st.file_uploader(
            "音声ファイルを選択してください (WAV, MP3, etc.)",
            type=["wav", "mp3", "m4a", "ogg", "flac"]
        )

        if uploaded_file is not None:
            with st.spinner("音声ファイルを処理中..."):
                waveform = process_uploaded_audio(uploaded_file)
            
            if waveform is not None:
                display_analysis_results(waveform, SAMPLE_RATE, models, sp)


if __name__ == "__main__":
    main()
