import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def is_meaningful(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    # Check if any word in the sentence is in the vocabulary
    meaningful_words = [word for word in sentence_words if word in words]
    
    # If at least one meaningful word is found, consider the sentence meaningful
    return len(meaningful_words) > 0


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

import re

def chatbot_response(msg):
    if is_meaningful(msg, words):
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    else:
        # If the input is not from the dataset, provide a generative response
        generative_responses = [
            "I'm not sure I understand. Could you please provide more details?",
            "Sorry, I didn't get that. Can you rephrase your question?",
            "It seems I'm not familiar with that. Can you ask something else?"            
        ]
        res = random.choice(generative_responses)
    
    return res



# #Creating GUI with tkinter
import tkinter
from tkinter import *


base = Tk()
base.title("The Pizza Shop ")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
base.configure(bg='#333')

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n')
        ChatLog.config(foreground="#fff", font=("Verdana", 12, "bold" ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 


#Create Chat window
ChatLog = Text(base, bd=0, bg="#333", height="10", width="50", font="Verdana",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base.configure(bg="#333"), command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",10,'bold'), text="Send", width="12", height="3",
                    bd=0, bg="#579822", activebackground="#4f8721",fg='#ffffff', activeforeground="#ffffff",
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="#333",width="29", height="3", font="Verdana", fg='#ffffff')
EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=5, y=401, height=90)

base.mainloop()


# import tkinter as tk
# from tkinter import Text, Scrollbar, Entry, Button, END

# # Your existing code here...

# # Create Chat window
# ChatLog = Text(base, bd=0, bg="white", height="10", width="50", font="Verdana")

# # Bind scrollbar to Chat window
# scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
# ChatLog['yscrollcommand'] = scrollbar.set

# # Create Button to send message
# SendButton = Button(base, font=("Verdana", 10, 'bold'), text="Send", width="12", height="3",
#                     bd=0, bg="#579822", activebackground="#4f8721", fg='#ffffff', activeforeground="#ffffff",
#                     command=send)

# # Create the box to enter message
# EntryBox = Entry(base, bd=0, bg="white", width="29", font="Verdana", fg='#000000')

# # Place all components on the screen
# scrollbar.place(x=376, y=6, height=386)
# ChatLog.place(x=6, y=6, height=386, width=370)
# EntryBox.place(x=128, y=401, height=90, width=265)
# SendButton.place(x=5, y=401, height=90)

# # Set GUI background to white
# base.configure(bg='white')

# # Calculate the center of the screen
# screen_width = base.winfo_screenwidth()
# screen_height = base.winfo_screenheight()
# x_coordinate = (screen_width / 2) - (400 / 2)
# y_coordinate = (screen_height / 2) - (500 / 2)

# # Set the GUI window to open in the center
# base.geometry(f"400x500+{int(x_coordinate)}+{int(y_coordinate)}")

# # Rest of your existing code...
# base.mainloop()

