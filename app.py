
import streamlit as st
import os
from audio_utils import AudioProcessor, save_audio_bytes
from spotify_client import SpotifyClient
from audio_recorder_streamlit import audio_recorder
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(page_title="感情に合わせて音楽を - Music Emotion Player", layout="wide", page_icon="🎵")

# --- Sidebar ---
with st.sidebar:
    st.title("Music Emotion Player")
    st.info("あなたの声から感情を読み取り、Spotifyから最適な音楽を提案します。")
    st.markdown("### 使い方")
    st.markdown("""
    1. **録音** または **ファイルアップロード** を選択
    2. 音声を入力して解析
    3. 検出された感情を確認（手動変更も可能）
    4. 提案されたプレイリストを試聴
    """)
    st.markdown("---")
    st.caption("Powered by OpenAI Whisper, HuBERT, & Spotify")
    
    # History Section
    st.markdown("---")
    st.subheader("📜 解析履歴")
    if st.session_state.get('history'):
        if st.button("履歴をクリア"):
            st.session_state.history = []
            st.rerun()
        
        for item in reversed(st.session_state.history):
            with st.expander(f"{item['time']} {item['emoji']} {item['emotion']}"):
                st.caption(f"ジャンル: {item['genre']}")
                st.write(f"♪ {item['track']}")
    else:
        st.caption("履歴はまだありません")

st.title("🎵 あなたの気分に合わせた音楽を")
st.write("マイクに向かって話しかけてください。あなたの声のトーンから感情を読み取り、ぴったりの音楽を選びます。")

# --- Initialize Models (Cached) ---
@st.cache_resource
def load_audio_processor():
    return AudioProcessor()

@st.cache_resource
def load_spotify_client():
    return SpotifyClient()

with st.spinner("AIモデルを読み込み中..."):
    processor = load_audio_processor()
    spotify = load_spotify_client()

# Use getattr to handle cases where the attribute might be missing (though removing cache should fix it)
auth_success = getattr(spotify, 'auth_success', False)

if not auth_success:
    st.error("Spotifyの認証に失敗しました。以下の点を確認してください：")
    st.markdown("""
    1. `.env` ファイルに `SPOTIPY_CLIENT_ID` と `SPOTIPY_CLIENT_SECRET` が正しく設定されているか。
    2. アプリケーション実行中に設定を変更した場合、**ターミナルで `Ctrl+C` を押してサーバーを停止し、再起動**してください（リロードだけでは反映されません）。
    """)
    st.stop()

# --- Application State ---
if 'emotion' not in st.session_state:
    st.session_state.emotion = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'processed_audio_bytes' not in st.session_state:
    st.session_state.processed_audio_bytes = None
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Audio Input Method ---
input_mode = st.radio("入力モードを選択", ["🎙️ マイクで録音", "📂 ファイルアップロード"], horizontal=True)

processed_audio = False # Flag to track if we processed new audio this run

if input_mode == "🎙️ マイクで録音":
    st.subheader("録音")
    recorded_audio_bytes = audio_recorder(
        text="クリックして録音開始",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x"
    )
    
    if recorded_audio_bytes:
        st.audio(recorded_audio_bytes, format="audio/wav")
        # Store for processing
        if st.session_state.processed_audio_bytes != recorded_audio_bytes:
            audio_bytes_to_process = recorded_audio_bytes
            processed_audio = True

elif input_mode == "📂 ファイルアップロード":
    st.subheader("音声ファイルをアップロード")
    uploaded_file = st.file_uploader("音声ファイルを選択してください (wav, mp3, m4a)", type=['wav', 'mp3', 'm4a'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("この音声を解析する"):
            # Read bytes from uploaded file
            file_bytes = uploaded_file.getvalue()
            if st.session_state.processed_audio_bytes != file_bytes:
                audio_bytes_to_process = file_bytes
                processed_audio = True
            else:
                 st.info("この音声は既に解析済みです。")

# --- Processing Logic ---
if processed_audio:
    with st.spinner("音声を解析中..."):
        # Determine file extension
        # Default to .wav for recording or if unknown
        file_ext = ".wav"
        if 'uploaded_file' in locals() and uploaded_file is not None:
             # Extract extension including dot, e.g. .mp3
            _, ext = os.path.splitext(uploaded_file.name)
            if ext:
                file_ext = ext
        
        temp_file = f"temp_input{file_ext}"
        save_audio_bytes(audio_bytes_to_process, temp_file)
        
        # 1. Transcribe
        transcription = processor.transcribe(temp_file)
        st.session_state.transcription = transcription if transcription else "（音声が検出されませんでした）"
        
        # 2. Emotion Analysis
        emotion_label = processor.predict_emotion(temp_file)
        # 3. Text Emotion Analysis
        text_emotion_label = processor.predict_text_emotion(transcription)
        
        # Combine logic: 
        # If Audio is Neutral but Text is Strong (Hap/Sad/Ang), use Text.
        # Otherwise respect Audio (tone often conveys more truth than text sarcasm, but for simple app, text content matters).
        final_emotion = emotion_label
        if emotion_label == 'neu' and text_emotion_label != 'neu':
            final_emotion = text_emotion_label
            
        st.session_state.emotion = final_emotion
        st.session_state.audio_emotion = emotion_label
        st.session_state.text_emotion = text_emotion_label
        
        # Increment analysis count to reset the selectbox state
        st.session_state.analysis_count += 1
        
        # Save processed bytes to prevent re-run
        st.session_state.processed_audio_bytes = audio_bytes_to_process
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

# --- Results Display ---
if st.session_state.transcription or st.session_state.emotion:
    st.divider()
    st.subheader("2. 解析結果")
    
    col1, col2 = st.columns(2)
    with col1:
        display_text = st.session_state.transcription if st.session_state.transcription else "（音声が検出されませんでした）"
        st.info(f"**認識されたテキスト:**\n\n{display_text}")
    
    with col2:
        # Translate emotion to Japanese and add visuals
        emotion_map = {
            'neu': {'label': 'ニュートラル (平常)', 'emoji': '😐', 'color': 'gray'},
            'hap': {'label': 'ハッピー (喜び)', 'emoji': '😄', 'color': 'green'},
            'sad': {'label': 'サッド (悲しみ)', 'emoji': '😢', 'color': 'blue'},
            'ang': {'label': 'アングリー (怒り)', 'emoji': '😠', 'color': 'red'}
        }
        
        # Retrieve separate emotions if available (backward compatibility)
        audio_em = getattr(st.session_state, 'audio_emotion', st.session_state.emotion)
        text_em = getattr(st.session_state, 'text_emotion', 'neu')
        
        # Determine current index for selectbox
        emotion_keys = list(emotion_map.keys())
        try:
            default_index = emotion_keys.index(st.session_state.emotion)
        except ValueError:
            default_index = 0
            
        # Display detected emotion with style
        final_info = emotion_map.get(st.session_state.emotion, {'label': st.session_state.emotion, 'emoji': '🤔'})
        
        # Detailed Breakdown
        st.markdown(f"""
        <div style="padding:10px; border-radius:10px; background-color:rgba(128,128,128,0.1); border-left: 5px solid {final_info.get('color', 'gray')};">
            <h4>総合判定: {final_info['emoji']} {final_info['label']}</h4>
            <small>音声分析: {emotion_map.get(audio_em, {}).get('emoji')} / テキスト分析: {emotion_map.get(text_em, {}).get('emoji')}</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer
        
        if 'genre' not in st.session_state:
            st.session_state.genre = 'All'
            
        # UI Layout for options
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            # Manual Override
            selected_emotion = st.selectbox(
                "感情を手動で変更・補正:",
                options=emotion_keys,
                format_func=lambda x: f"{emotion_map[x]['emoji']} {emotion_map[x]['label']}",
                index=default_index,
                key=f"emotion_select_{st.session_state.analysis_count}"
            )
            
        with opt_col2:
            # Genre Selector
            genre_options = ['All', 'J-Pop', 'K-Pop', 'Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Lo-Fi', 'Classical', 'Electronic']
            selected_genre = st.selectbox(
                "ジャンルで絞り込み:",
                options=genre_options,
                index=genre_options.index(st.session_state.genre) if st.session_state.genre in genre_options else 0,
                key=f"genre_select_{st.session_state.analysis_count}"
            )
            # Update state (though redundant with key usually, simple assignment helps clarify intent if used elsewhere)
            st.session_state.genre = selected_genre

# --- Spotify Recommendations ---
if st.session_state.emotion:
    st.divider()
    
    # Use selected_emotion (from dropdown) instead of raw detected emotion
    current_mood_key = selected_emotion if 'selected_emotion' in locals() else st.session_state.emotion
    # Use selected genre
    current_genre = selected_genre if 'selected_genre' in locals() else 'All'
    
    # Get mood info for header
    # Re-define map if needed or access from above block (but scope might differ if we extracted function, here it's fine)
    emotion_map = {
        'neu': {'label': 'ニュートラル (平常)', 'emoji': '😐'},
        'hap': {'label': 'ハッピー (喜び)', 'emoji': '😄'},
        'sad': {'label': 'サッド (悲しみ)', 'emoji': '😢'},
        'ang': {'label': 'アングリー (怒り)', 'emoji': '😠'}
    }
    mood_info = emotion_map.get(current_mood_key, {'label': current_mood_key, 'emoji': '🎵'})
    
    genre_text = f"({current_genre})" if current_genre != "All" else ""
    st.subheader(f"3. {mood_info['emoji']} {mood_info['label']} 気分のあなたへのおすすめ {genre_text}")
    
    recommendations = spotify.get_recommendations(current_mood_key, genre=current_genre)
    
    if isinstance(recommendations, dict) and "error" in recommendations:
        st.error(f"Spotify API Error: {recommendations['error']}")
    elif recommendations:
        for track in recommendations:
            track_id = track.get('id')
            if track_id:
                # Embed Player
                components.iframe(f"https://open.spotify.com/embed/track/{track_id}", height=80)
            else:
                # Fallback to link if no ID (shouldn't happen with valid response)
                st.markdown(f"[{track['name']} - {track['artist']}]({track['url']})")
                
        # Update History
        if recommendations:
            top_track = recommendations[0]
            timestamp = datetime.now().strftime("%H:%M")
            new_entry = {
                "time": timestamp,
                "emoji": mood_info['emoji'],
                "emotion": mood_info['label'],
                "genre": current_genre,
                "track": f"{top_track['name']} - {top_track['artist']}"
            }
            
            # Avoid adding duplicate if it matches the last entry (ignoring time)
            should_add = True
            if st.session_state.history:
                last = st.session_state.history[-1]
                if (last['emotion'] == new_entry['emotion'] and 
                    last['genre'] == new_entry['genre'] and 
                    last['track'] == new_entry['track']):
                    should_add = False
            
            if should_add:
                st.session_state.history.append(new_entry)
                
    else:
        st.warning("音楽が見つかりませんでした。Spotifyの認証設定または再生可能な曲が見つかりません。")
