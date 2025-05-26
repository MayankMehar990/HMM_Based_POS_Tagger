# POS_Tagger

This is a POS Tagger i.e. an ML program that tags a sentence with POS(Parts of speech) such as noun, verb etc. It utilizes the Hidden Markov Model and Conditional Random Fields model. The first Model is trained on the universal tagset dataset available in the brown corpus in nltk module of python, while second one also incorporates the punkt dataset in addition to the brown and universal tagset dataset. This project is created by Mayank Mehar as a part of team project. The contributers of this project are Jayant, Kaushikraj and Raunak.

The HMM_pos_tagging.py file contains the implementation of the HMM and it trains a ML model on the whole universal-tagset dataset.

The HMM_Tagger_Results.py file contains the implementation of the HMM, it divides the dataset in 5 parts and trains the ML model on 4 parts of the dataset and gauges it performance by testing it on the 5th one. It does so iterativerly for all 5 parts. This is known as 5-fold cross validation. It then outputs the average performance of the 5 tests.

The CRF_pos_Tagging.py file implements the CRF model trained on the whole dataset which is combined from the three aforementioned datasets.

The CRF_Tagger_Results.py contains the performance of the crf_model which is again tested using a 5-fold cross validation method. This file outputs the performance metrics of the crf model.

The GUI.py file implements a  basic GUI interface using the tkinter module. It uses the models trained in the HMM_pos_tagging.py and CRF_pos_Tagging.py files to tag the sentence given by the user in the input field. It consists if 3 buttons, a HMM_Tagger, a CRF_Tagger and a close button. User can choose to tag a sentence using a hmm or crf based model.
