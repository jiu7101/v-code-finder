import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.utils import which
import tempfile
import subprocess
import os

AudioSegment.converter = which("ffmpeg")

st.title("🎙️ V-Code Finder")
st.subheader("당신의 목소리는 어떤 계절인가요?")
st.markdown("음성 파일을 업로드하면, 목소리의 특징을 분석해 계절 유형을 알려드릴게요!")

uploaded_file = st.file_uploader("🎧 음성 파일(mp3, wav, m4a)을 업로드하세요", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    suffix = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in.flush()

        if suffix == "m4a":
            # ffmpeg를 직접 호출해 m4a → wav 변환
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
                command = [
                    "ffmpeg", "-i", tmp_in.name,
                    "-ac", "1", "-ar", "22050",
                    "-t", "5",  # 5초까지만
                    tmp_out.name
                ]
                try:
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    y, sr = librosa.load(tmp_out.name)
                except Exception as e:
                    st.error(f"m4a 처리 중 오류 발생: {e}")
                    st.stop()
        else:
            # mp3/wav 직접 처리
            try:
                audio = AudioSegment.from_file(tmp_in.name, format=suffix)
            except Exception as e:
                st.error(f"파일 처리 오류: {e}")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                audio = audio.set_channels(1).set_frame_rate(22050)
                audio = audio[:5000]
                audio.export(tmp_wav.name, format="wav")
                y, sr = librosa.load(tmp_wav.name)

    # 분석
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid_pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch = valid_pitches.mean() if valid_pitches.size > 0 else 0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = np.sum(y ** 2) / len(y)

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

    st.markdown("---")
    st.success(result_dict[season]["title"])
    st.write(result_dict[season]["desc"])
    st.markdown("---")
    st.markdown("🔍 더 정밀한 분석이 필요하다면? **Speech Code 전문가 진단**을 추천드려요.")
    st.caption(f"📊 분석 수치 → Pitch: {pitch:.2f}, Tempo: {tempo:.2f}, Energy: {energy:.5f}")
