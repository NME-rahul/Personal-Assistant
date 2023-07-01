import model_eva
from tensorflow.keras.models import load_model

import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

import time
import os
import sys
import threading

class Bot:
    def __init__(self, user, accent):
        self.accent = accent; self.user = user
        gTTS(f"   Hello {user}, How can i help you", lang='en', tld=accent).save("Hello.mp3")
        print(f"Eva: Hello! {user}, How can i help you.\n")
        playsound("Hello.mp3")
        
    def speech_to_text(self):
        try:
            r = sr.Recognizer()
            r.pause_threshold = 1
            with sr.Microphone(device_index = 1) as source:
                print("listening...")
                audio = r.listen(source)
            text = r.recognize_sphnix(audio)
        except sr.UnknownValueError:
            print("Unable to understand, say again."); text = False
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e)); text = False

        return text

    def text_to_speech(self):
        speech = gTTS(text, lang='en', tld=self.accent)
        speech.save("speech.mp3")
        playsound("speech.mp3")
        os.system("rm speech.mp3")

    def load():
        return load_model('C:/Users/arvin/OneDrive/Desktop/data/eva_model.h5')        
        
         
if __name__ == "__main__" :
    user = "Rahul"; accent = "ca"
    if len(sys.argv) == 2:
        user = sys.argv[1]; accent = "ca";
    elif len(sys.argv) == 3:
        user = sys.argv[1]; accent = sys.argv[2]
    if 'train' in sys.argv:
        pass

    eva = Bot(user, accent)
    cnt = 0
    model = eva.load()
    tokenizer = create_sequence(load_data())
    
    while True:
        text = eva.speech_to_text()
        print(text)
        if text != False:
            text = model_eva.generate_text(model, tokenizer, text, next_word=20)
            eva.text_to_speech(text)
        else:
            cnt = cnt + 1
            time.sleep(3)
            if cnt == 3:
                sys.exit(0)

