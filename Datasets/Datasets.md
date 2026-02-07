Dataset Title:-
EdNET KT-4

Usage of Dataset:-
The EdNet-KT4 dataset was used as the primary data source to model and analyze student learning behavior in an expert-agnostic Intelligent Tutoring System,Student interaction logs from the dataset were utilized to capture temporal sequences, behavioral patterns, and performance outcomes without relying on expert-labeled annotations.
The dataset was filtered and preprocessed to form fixed-length interaction sequences, which were then used to train and evaluate multiple deep learning models under a self-supervised learning setup. EdNet-KT4 enabled realistic experimentation due to its large scale and rich interaction features, making it suitable for assessing the scalability, robustness, and generalization capability of the proposed tutoring framework.

Dataset Information:-
The study uses the EdNet-KT4 dataset, a large-scale educational dataset consisting of real student interaction logs collected from an online tutoring system . It contains data from over 300,000 learners, with each user’s interactions stored as sequential records.The dataset includes key attributes such as timestamps, action types, item IDs, time spent (cursor time), content source, platform details, and binary correctness of responses. These rich temporal and behavioral features make EdNet-KT4 well suited for self-supervised learning, enabling effective modeling of student performance and learning behavior without requiring expert-labeled data.

Dataset Name:-
EdNet-KT4 Dataset 

Source:-
EdNet dataset collected from the Santa online tutoring system (EdNet-KT4)

Domain:-

Education / Intelligent Tutoring Systems / Learning Analytics

Task:-
Predict the next student interaction correctness and model learner behavior from past interaction sequences


Problem Type:-

Binary classification (Correct vs. Incorrect) using self-supervised sequential learning



File Format:-
Individual CSV files per user containing sequential interaction logs


Dataset Link:
http://bit.ly/ednet-kt4


**Dataset Overview**



Total Records:-
Over 300,000 learner interaction sequences collected from the EdNet-KT4 dataset


Labeled Records:-
All records contain implicit labels derived from student responses (correct = 1, incorrect = 0); no manual expert labeling is used


Classes:-
Two classes – Correct and Incorrect


Annotation Type:-
Self-supervised / implicit annotation, automatically obtained from student interaction logs


Why This Dataset?

EdNet-KT4 provides large-scale, real-world educational interaction data with rich temporal and behavioral features, making it ideal for expert-agnostic, self-supervised learning and scalable Intelligent Tutoring System research.


Features Used:-
Feature 1:Timestamp – captures the chronological order of student interactions

Feature 2:Action Type – indicates the type of activity performed (e.g., reading, answering)

Feature 3:Item ID – identifies the learning content or question attempted

Feature 4:Cursor Time – represents the time spent on a specific learning item

Feature 5:Source – denotes the content delivery channel

Feature 6:Platform – specifies the device used (desktop or mobile)

Feature 7:User Answer – binary indicator of response correctness (correct/incorrect)

Summary:
The EdNet-KT4 dataset provides large-scale, real-world student interaction data with rich temporal, behavioral, and performance-related features. It enables effective self-supervised learning for modeling student behavior and predicting performance without expert annotations, making it well suited for building scalable and expert-agnostic Intelligent Tutoring Systems.





