from pydub import AudioSegment
import speech_recognition as sr

# MP3 파일을 WAV로 변환
audio = AudioSegment.from_mp3("단, 한개의 지표로 3년 만에 1,200억 만든 매매법(손익비 초고수) (128 kbps).mp3")
audio.export("단, 한개의 지표로 3년 만에 1,200억 만든 매매법(손익비 초고수) (128 kbps).wav", format="wav")

# 음성 인식기 초기화
recognizer = sr.Recognizer()

# 변환된 WAV 파일을 읽고 텍스트로 변환
with sr.AudioFile("단, 한개의 지표로 3년 만에 1,200억 만든 매매법(손익비 초고수) (128 kbps).wav") as source:
    audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data, language='ko-KR')  # 한국어 인식을 위해 'ko-KR' 설정

# 결과 출력
print(text)

