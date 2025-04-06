import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.utils import which
import tempfile

# 🔧 ffmpeg 경로 명시 (m4a 처리 안정화)
AudioSegment.converter = which("ffmpeg")

# 제목
st.title("🎙️ V-Code Finder")
st.subheader("당신의 목소리는 어떤 계절인가요?")
st.markdown("음성 파일을 업로드하면, 목소리의 특징을 분석해 계절 유형을 알려드릴게요!")

# 파일 업로드
uploaded_file = st.file_uploader("🎧 음성 파일(mp3, wav, m4a)을 업로드하세요", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    file_suffix = uploaded_file.name.split('.')[-1]

    # 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_suffix}") as tmp_raw_file:
        tmp_raw_file.write(uploaded_file.read())
        tmp_raw_file.flush()

        # 오디오 로딩
        try:
            audio = AudioSegment.from_file(tmp_raw_file.name, format=file_suffix)
        except Exception as e:
            st.error(f"파일을 처리하는 중 오류가 발생했어요: {e}")
            st.stop()

        # wav 변환용 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
            audio = audio.set_channels(1).set_frame_rate(22050)
            audio = audio[:5000]  # 앞 5초만 사용
            audio.export(tmp_wav_file.name, format="wav")
            y, sr = librosa.load(tmp_wav_file.name)

    # 분석
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid_pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch = valid_pitches.mean() if valid_pitches.size > 0 else 0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = np.sum(y ** 2) / len(y)

    # 분류 기준
    def classify_voice(pitch, tempo, energy):
        if pitch > 180 and tempo > 100 and energy < 0.01:
            return "봄"
        elif pitch > 180 and tempo <= 100:
            return "여름"
        elif pitch <= 180 and energy > 0.02:
            return "겨울"
        else:
            return "가을"

    season = classify_voice(pitch, tempo, energy)

    # 결과 텍스트
    result_dict = {
        "봄": {
            "title": "☀️ 당신의 Voice Type은 [봄]입니다.",
            "desc": "밝고 경쾌한 말투로 분위기를 환하게 만드는 스타일이에요.\n대표 인물: 유인나, 박나래, 하하",
        },
        "여름": {
            "title": "🌊 당신의 Voice Type은 [여름]입니다.",
            "desc": "자연스럽고 감각적인 말투로 편안한 분위기를 만들어내는 스타일이에요.\n대표 인물: 유재석, 장도연, 이이경",
        },
        "가을": {
            "title": "🍂 당신의 Voice Type은 [가을]입니다.",
            "desc": "따뜻하고 안정적인 말투로 신뢰를 주는 스타일이에요.\n대표 인물: 아이유, 전현무, 이서진",
        },
        "겨울": {
            "title": "❄️ 당신의 Voice Type은 [겨울]입니다.",
            "desc": "또렷하고 단단한 말투로 카리스마를 드러내는 스타일이에요.\n대표 인물: 김연아, 김혜수",
        },
    }

    # 출력
    st.markdown("---")
    st.success(result_dict[season]["title"])
    st.write(result_dict[season]["desc"])
    st.markdown("---")
    st.markdown("🔍 더 정밀한 분석이 필요하다면? **Speech Code 전문가 진단**을 추천드려요.")
    st.caption(f"📊 분석 수치 → Pitch: {pitch:.2f}, Tempo: {tempo:.2f}, Energy: {energy:.5f}")
