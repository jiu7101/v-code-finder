import streamlit as st
import librosa
import numpy as np
import tempfile
import subprocess
import os

st.title("🎙️ V-Code Finder")
st.subheader("당신의 목소리는 어떤 계절인가요?")
st.markdown("mp3, wav, m4a 파일을 업로드하면, 목소리의 특징을 분석해 계절 유형을 알려드릴게요!")

uploaded_file = st.file_uploader("🎧 음성 파일을 업로드하세요", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    suffix = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_input:
        tmp_input.write(uploaded_file.read())
        tmp_input.flush()

        # ffmpeg로 wav 변환
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_output:
            command = [
                "ffmpeg", "-y", "-i", tmp_input.name,
                "-ac", "1", "-ar", "22050",
                "-t", "5",  # 앞 5초만
                tmp_output.name
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                y, sr = librosa.load(tmp_output.name)
            except Exception as e:
                st.error(f"❌ ffmpeg 변환 실패: {e}")
                st.stop()

    if len(y) == 0:
        st.error("❌ 변환된 오디오가 비어 있어요.")
        st.stop()

    # 음성 분석
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid_pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch = valid_pitches.mean() if valid_pitches.size > 0 else 0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = np.sum(y ** 2) / len(y)

    # 계절 분류
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

    # 결과
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

