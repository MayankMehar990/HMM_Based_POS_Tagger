from tkinter import *
from HMM_pos_tagging import hmm_model
from CRF_pos_Tagging import crf_model

root = Tk()
root.title("POS Tagger")
root.geometry("560x400")


label1 = Label(root, text="Enter a sentence : ",width=15,font=("Times New Roman",20))
label1.grid(row=0, column=0)

entry1 = Text(root,width=30,height=5)
entry1.grid(row=0, column=1,pady=15)

label2 = Label(root, text="POS Tags : ",width=15,font=("Times New Roman",20))
label2.grid(row=1, column=0)

output_box = Text(root,width=30,height=8,state='disabled')
output_box.grid(row=1,column=1)

def hmm_Tagging(sentence):
    sentence=sentence.split()
    tagged_sentence=hmm_model.viterbi(sentence)
    return " ".join(f"{word}/{tag} " for word, tag in tagged_sentence)


def crf_Tagging(sentence):
    sentence=sentence.split()
    tagged_sentence=crf_model.predict(sentence)
    return " ".join(f"{word}/{tag} " for word, tag in zip(sentence,tagged_sentence))

def hmm_clicked():
    label1.configure(text="Entered Sentence : ")
    sentence=entry1.get("1.0",END).strip()
    tagged_sentence=hmm_Tagging(sentence)
    output_box.config(state='normal')  
    output_box.delete("1.0", END) 
    output_box.insert("1.0",tagged_sentence)     
    output_box.config(state='disabled')

def crf_clicked():
    label1.configure(text="Entered Sentence : ")
    sentence=entry1.get("1.0",END).strip()
    tagged_sentence=crf_Tagging(sentence)
    output_box.config(state='normal')  
    output_box.delete("1.0", END) 
    output_box.insert("1.0",tagged_sentence)     
    output_box.config(state='disabled')
    

button1 = Button(root, text="HMM_Tagger",command=hmm_clicked,width=10,font=("Times New Roman",15))
button1.grid(row=0,column=1,pady=40)
button1.place(x=250,y=290)

button1 = Button(root, text="CRF_Tagger",command=crf_clicked,width=10,font=("Times New Roman",15))
button1.grid(row=0,column=1,pady=40)
button1.place(x=400,y=290)

close=Button(root, text="Close", command=root.destroy,width=10,font=("Times New Roman",15))
close.grid(row=0,column=2,pady=40)
close.place(x=320,y=340)

root.mainloop()
