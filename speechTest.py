import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Capture audio from the default microphone
with sr.Microphone() as source:
    print("Adjusting for ambient noise. Please wait...")
    recognizer.adjust_for_ambient_noise(source, duration=5)
    print("Listening...")
    audio = recognizer.listen(source)

# Recognize speech using Google Web Speech API
try:
    print("Recognizing...")
    text = recognizer.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Google Web Speech API could not understand the audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Web Speech API; {e}")
