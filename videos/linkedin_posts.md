# LinkedIn Posts — CodeAlpha ML Internship (Professional Tone)
Student: Shadman Samin | ID: CA/DF1/298776 | Always tag @CodeAlpha

---

## STATUS POST (publish first)
I am pleased to share that I have started my Machine Learning internship @CodeAlpha (September–October 2026 batch).

Over the coming weeks, I will be building and documenting four applied ML projects: credit scoring, speech emotion recognition, handwritten character recognition, and disease prediction — with full source code on GitHub and video walkthroughs to follow.

Grateful for the opportunity and looking forward to sharing progress.

#MachineLearning #CodeAlpha #Internship

---

## POST 1 — Credit Scoring (attach F:\codeAlpha\videos\Video1-Credit-Scoring.mp4)
CodeAlpha ML Internship — Project 1 of 4 @CodeAlpha

**Credit Scoring Model.** Predicting loan creditworthiness on the German Credit dataset (1,000 customers, 20 financial features; 70% good / 30% bad).

Approach: one-hot + scaling pipeline; class-balanced Logistic Regression, Decision Tree, and Random Forest with 5-fold cross-validation.
Result: Random Forest achieves 72% accuracy with ROC-AUC 0.78.

The one-minute video below walks through the data, method, and results. Full source code, plots, and metrics:
https://github.com/Shadman-Samin/CodeAlpha/tree/main/Task1-Credit-Scoring

#MachineLearning #CodeAlpha #Python #ScikitLearn

---

## POST 2 — Emotion Recognition (attach F:\codeAlpha\videos\Video2-Emotion-Speech.mp4)
Project 2 of 4 @CodeAlpha

**Speech Emotion Recognition.** Classifying happy, angry, sad, and neutral speech per the RAVDESS/TESS standard.

Approach: 40 MFCC features extracted with Librosa; deep neural network in PyTorch (CNN/LSTM-ready pipeline; real datasets drop into the data folder with no code change).
Result: verified end-to-end on 480 clips, confirming the pipeline ahead of noisy real-world audio.

Video walkthrough below. Code and confusion matrix:
https://github.com/Shadman-Samin/CodeAlpha/tree/main/Task2-Emotion-Speech

#DeepLearning #PyTorch #CodeAlpha #MachineLearning

---

## POST 3 — Handwritten Recognition (attach F:\codeAlpha\videos\Video3-Handwritten-Recognition.mp4)
Project 3 of 4 @CodeAlpha

**Handwritten Character Recognition.** MNIST digit classification (60,000 train / 10,000 test, 28×28 grayscale).

Approach: compact CNN — two convolutional layers, max-pooling, dropout, Adam optimizer, GPU-trained.
Result: 98.87% test accuracy after 3 epochs; extensible to EMNIST letters and CRNN word models.

One-minute video below with sample predictions. Code:
https://github.com/Shadman-Samin/CodeAlpha/tree/main/Task3-Handwritten-Recognition

#ComputerVision #DeepLearning #CodeAlpha

---

## POST 4 — Disease Prediction (attach F:\codeAlpha\videos\Video4-Disease-Prediction.mp4)
Project 4 of 4 — all CodeAlpha ML tasks complete @CodeAlpha

**Disease Prediction from Medical Data.** Breast Cancer Wisconsin (569 patients, 30 measurements) plus binarized Diabetes progression.

Approach: standardized pipeline comparing Logistic Regression, SVM, Random Forest (300 trees), and XGBoost with stratified splits.
Results: breast cancer AUC 0.995; diabetes accuracy 75%. All confusion matrices and metrics published.

Final video walkthrough below. Full comparison and code:
https://github.com/Shadman-Samin/CodeAlpha/tree/main/Task4-Disease-Prediction

Thank you to Team CodeAlpha for the mentorship and structured program. All four projects, code, and documentation are public in one repository:
https://github.com/Shadman-Samin/CodeAlpha

#MachineLearning #HealthcareAI #CodeAlpha
