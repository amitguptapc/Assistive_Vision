from tensorflow.keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
import pickle
import numpy as np
import random
import tensorflow as tf
max_len = 74

# loading the models
model = load_model('./features/model.h5')
feature_extractor = load_model("./features/feature_extractor.h5")
with open('./features/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('./features/idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)
    
# function to extract features from the input image    
def encode_image(img):
    img = image.load_img(img,target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    feature_vector = feature_extractor.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector    

# function to predict the caption using Beam Search
def predict_caption_beamsearch(img, beam_index = 3):
    start = [word2idx["<BEGIN>"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encode_image(img)
            preds = model.predict([np.array([e]), np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]
    final_caption = []
    for i in intermediate_caption:
        if i != '<END>':
            final_caption.append(i)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption

# function to predict caption using Greedy Sampling
def predict_caption_greedy(img):
    start_word = ["<BEGIN>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encode_image(img)
        preds = model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        if word_pred == "<END>" or len(start_word) > max_len:
            break
    return ' '.join(start_word[1:-1])
