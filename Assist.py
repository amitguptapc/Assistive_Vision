from gtts import gTTS
from playsound import playsound
import Caption_it as ct
caption = ct.caption_this_image('./sample/shore.jpg')
print(caption)
speech = gTTS(caption)
speech.save('Audio/speech.wav')
playsound('./Audio/speech.wav')
