# HMM_Based_POS_Tagger

This is a POS Tagger i.e. an ML program that tags a sentence with POS(Parts of speech) such as noun, verb etc. It utilizes the Hidden Markov Model. This Model is trained on the universal tagset dataset available in the brown corpus in nltk module of python. This project is created by Mayank Mehar as a part of team project. The contributers of this project are Jayant, Kaushikraj and Raunak.

The HMM_pos_tagging.py file contains the implementation of the HMM and it trains a ML model on the whole dataset.

The HMM_Tagger_Results.py file contains the implementation of the HMM, it divides the dataset in 5 parts and trains the ML model on 4 parts of the dataset and gauges it performance by testing it on the 5th one. It does so iterativerly for all 5 parts. This is known as 5-fold cross validation. It then outputs the average performance of the 5 tests.

The GUI.py file implements a  basic GUI interface using the tkinter module. It uses the model trained in the HMM_pos_tagging.py file to tag the sentence given by the user in the input field.
