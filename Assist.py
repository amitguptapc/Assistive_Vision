from gtts import gTTS
from playsound import playsound
import Caption_it as ct
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn')


n = 0
plt.ion()
while n<5:
    cap = cv2.VideoCapture(0)
    ret,img = cap.read()
    if ret == False:
        continue   
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.imsave("./sample/image{}.jpg".format(n),img)
    
    caption = ct.predict_caption_greedy("sample/image{}.jpg".format(n))
    plt.title(caption)
    
    print(caption)
    speech = gTTS(caption)
    speech.save('./audio/speech.wav')
    playsound('./audio/speech.wav')
    n += 1
plt.show()