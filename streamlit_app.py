import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
import os
import tempfile

# 제목
st.title("🎙️ V-Code Finder")
st.subheader("당신의 목소리는 어떤 계절인가요?")
st.markdown("음성 파일을 업로드하면, 목소리의 특징을 분석해 계절 유형을 알려드릴게요!")

# 파일 업로드
uploaded_file = st.file_uploader("🎧 음성 파일(mp3, wav, m4a)을 업로드하세요", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        # 파일 형식 분기 처리
        if uploaded_file.type == "audio/mp3":
            audio = AudioSegment.from_file(uploaded_file, format="mp3")
        elif uploaded_file.type == "audio/x-m4a" or uploaded_file.type == "audio/m4a":
            audio = AudioSegment.from_file(uploaded_file, format="m4a")
        else:
            audio = AudioSegment.from_file(uploaded_file, format="wav")

        # 오디오 처리
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio = audio[:5000]  # 앞 5초만 사용
        audio.export(tmp_wav_file.name, format="wav")

        # librosa 로드
        y, sr = librosa.load(tmp_wav_file.name)

        # 분석
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[magnitudes > np.median(magnitudes)].mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = np.sum(y ** 2) / len(y)

        # 결과 분류 기준
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

        # 결과 문구
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

        # 결과 출력
        st.markdown("---")
        st.success(result_dict[season]["title"])
        st.write(result_dict[season]["desc"])
        st.markdown("---")
        st.markdown("🔍 더 정밀한 분석이 필요하다면? Speech Code 전문가 진단을 추천드려요.")
