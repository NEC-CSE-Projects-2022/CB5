
# Team Number – Project Title

## Team Info
- 22471A05K3 — **Name** ( [LinkedIn](https://www.linkedin.com/in/tippanaboina-ramesh-311b5024b/) )
_Work Done: Research and finding resources, train the model, optimization of results


- 22471A05J0 — **Name** ( [LinkedIn](https://www.linkedin.com/in/shaik-fayaz-291912277) )
_Work Done:preprocessing of dataset, labelling


- 22471A05K9 — **Name** ( [LinkedIn](https://www.linkedin.com/in/jakraiah-kinnera-9665892b7) )
_Work Done: evaluation of results and verification

---

## Abstract
Expert-tailored annotations and domain-specific
rules are usually unavoidable in traditional Intelligent Tutoring
Systems (ITS), restricting scalability and flexibility. This paper
presents a new expert-agnostic approach to intelligent tutoring
based on self-supervised learning to promote more personalized
education with-out depending on domain experts. We develop
and evaluate multiple deep learning models—GRU, BiLSTM,
LSTM, CNN, Transformer, MLP, and hybrid Embedded GRU
CNN—trained on student interaction datasets using automatic
representation learning techniques. Our approach leverages the
sequential nature of learning behaviours and embeds contextu
alized features to identify optimal learning interventions. Among
the tested architectures, Embedded GRU-CNN and BiLSTM,
CNN models demonstrated superior accuracy (up to 99%) in
predicting learner needs and engagement levels. The findings
demonstrate substantial student modelling performance improve
ment without hand-crafted labels, affirming the promise of self
supervised methods in ITS. The work opens to scalable, domain
agnostic intelligent tutoring systems that can adapt and provide
feedback in real time, a step toward democratizing AI-facilitated
education for multiple types of learners.

---

## Paper Reference (Inspiration)
👉 **[Paper Title :-Expert-Agnostic AI for Intelligent Tutoring
Systems: Leveraging Self-Supervised Knowledge
Mining
  – Author Names :-S.V.N. Sreenivasu, Tippanaboina Ramesh, Shaik Mohammad Fayaz, Kinnera Yarra Jakraiah,
Dharmapuri Siri, Sesha Bhargavi Velagaleti,
 ](Paper URL here)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Our work introduces an expert-agnostic, self-supervised ITS framework that learns directly from raw student interaction data without manual annotations.
Using a hybrid GRU-CNN architecture, it achieves higher accuracy and better generalization than traditional expert-driven and single-model approaches.


---

## About the Project
This project develops an expert-agnostic Intelligent Tutoring System (ITS) that uses self-supervised deep learning to understand and predict student learning behavior. Instead of relying on manually labeled data or predefined teaching rules, the system learns directly from raw student interaction logs, such as question attempts, time spent on activities, correctness of answers, and learning sequences. By analyzing these interactions, the system can model how students learn over time and identify patterns related to engagement and performance.
Most traditional tutoring systems require continuous involvement of subject experts and are difficult to scale across subjects or large numbers of learners. This project overcomes those limitations by eliminating expert dependency, making the system scalable, flexible, and cost-effective. It supports personalized learning, adapts to different learner behaviors, and is especially useful for large online education platforms and low-resource learning environments where expert intervention is limited.

General Project Workflow :

1.Input: Raw student interaction data, including timestamps, action types, question identifiers, time spent on tasks, correctness of responses, and platform information.

2.Processing: Data cleaning, normalization, feature engineering, and conversion of interactions into fixed-length sequences to capture temporal learning behavior.

3.Model: Self-supervised deep learning models such as MLP, LSTM, BiLSTM, GRU, CNN, Transformer, and a hybrid GRU-CNN, which learn both short-term and long-term learning patterns.
4.Output: Accurate predictions of student performance and engagement, which can be used to provide adaptive feedback, learning recommendations, and improved tutoring support in real time.


---

## Dataset Used
👉 **[EdNET KT4](http://bit.ly/ednet-kt4)**

**Dataset Details:**
The project uses the EdNet-KT4 dataset, a large-scale educational dataset containing real student interaction logs collected from an online tutoring platform. It includes data from hundreds of thousands of learners, making it suitable for building scalable and realistic Intelligent Tutoring Systems.
Each student’s data is stored as a sequence of interactions and contains rich temporal, behavioral, and performance-related information. Key attributes include timestamps of actions, type of activity performed (such as reading or answering questions), question or content identifiers, time spent on each item, platform used (mobile or desktop), and whether the student’s answer was correct or incorrect.
For this project, a filtered and cleaned subset of the dataset is used to ensure data quality and manageable computation. The dataset is processed into fixed-length interaction sequences, enabling deep learning models to capture learning progression, engagement patterns, and performance trends without requiring any manual labels or expert annotations. This makes the dataset ideal for self-supervised, expert-agnostic learning.


---

## Dependencies Used
Data Handling & Preprocessing:
pandas – loading, cleaning, and filtering EdNet-KT4 student interaction data
numpy – numerical operations and tensor preparation
scikit-learn –
Label encoding for categorical features
Train–test split
Feature scaling
Evaluation metrics (accuracy, precision, recall, F1-score)
Deep Learning Models:
TensorFlow / Keras – implementation of:
MLP,LSTM,BiLSTM,GRU,CNN,Transformer
Hybrid GRU-CNN (core model)
Model Training & Optimization:
Adam Optimizer – model optimization
Binary Cross-Entropy Loss – performance prediction
Dropout & Early Stopping – overfitting prevention
Evaluation & Visualization:
scikit-learn metrics – accuracy, precision, recall, F1-score
matplotlib – training/testing loss curves and performance plots

---

## EDA & Preprocessing
1.Exploratory Data Analysis (EDA):-Exploratory Data Analysis was performed on the EdNet-KT4 dataset to understand student interaction patterns, data distribution, and sequence characteristics. Initial analysis examined the frequency of user interactions, correctness distribution, time spent on learning items, and platform usage to identify behavioral trends and inconsistencies. Sessions with very few interactions were removed to reduce noise, and missing or malformed records were discarded to ensure data reliability.
Preprocessing Steps (Step-by-Step):-
Data Loading:
The EdNet-KT4 dataset is loaded using pandas, where each learner’s interaction data is read from CSV files.
Session Filtering:
Student sessions with fewer than a minimum number of interactions are removed to eliminate noisy and non-informative learning sequences.
Missing Value Handling:
Rows containing missing, null, or invalid values are dropped to ensure data consistency and reliability.
Data Cleaning:
Records with incorrect timestamps or malformed item identifiers are identified and removed.
Categorical Encoding:
Categorical features such as action type, item ID, source, and platform are converted into numerical values using label encoding.
Numerical Scaling:
Continuous features like cursor time are normalized to bring all numerical inputs to a comparable scale.
Temporal Feature Engineering:
Time differences between consecutive interactions are calculated to capture learning pace and temporal behavior.
Behavioral Feature Engineering:
Rolling averages of correctness and time spent are computed to represent short-term performance trends.
Sequence Construction:
Student interaction data is organized into fixed-length sequences, applying padding or truncation as needed.
Tensor Conversion:
The final processed data is transformed into 3D tensors suitable for input into deep learning models.

---

## Model Training Info
All models in this research were trained using a self-supervised learning approach, where the objective was to predict the next student interaction’s correctness based on a sequence of past interactions. The processed EdNet-KT4 data was split into training and testing sets, and interactions were organized into fixed-length sequences of 100 steps. Training was carried out using the Adam optimizer with Binary Cross-Entropy loss to ensure efficient convergence. To improve generalization and reduce overfitting, dropout layers and early stopping were applied during training. A batch size of 64 was used, and hidden layer sizes ranged from 128 to 256 units depending on the model architecture. Training and validation loss curves were monitored to analyze convergence behavior and model stability across different deep learning architectures 


---

## Model Testing / Evaluation
Model evaluation was conducted on a held-out test set from the EdNet-KT4 dataset to rigorously assess the effectiveness and reliability of the proposed expert-agnostic Intelligent Tutoring System, After training, each deep learning model—MLP, LSTM, BiLSTM, GRU, CNN, Transformer, and the hybrid GRU-CNN—was evaluated on unseen student interaction sequences to ensure fair comparison and to test real-world generalization capability.
Standard classification metrics, including Accuracy, Precision, Recall, and F1-score, were used to capture multiple aspects of performance. While accuracy measured overall correctness, precision and recall evaluated the models’ ability to reduce false positives and false negatives, and the F1-score provided a balanced assessment of predictive quality. In addition to metric-based evaluation, training and testing loss curves were closely analyzed to study convergence behavior, detect overfitting or underfitting, and understand learning stability across different architectures.
Comparative analysis revealed clear performance differences among models. Sequential and hybrid models significantly outperformed non-sequential baselines, highlighting the importance of temporal learning in student behavior modeling. In particular, CNN and hybrid GRU-CNN models achieved the highest accuracy (up to 98.41%), exhibiting stable loss convergence and minimal train–test divergence. These results demonstrate that combining local pattern extraction with temporal dependency modeling leads to robust, scalable, and highly generalizable performance, validating the effectiveness of self-supervised, expert-agnostic approaches for intelligent tutoring systems.

---

## Results
The results show that the proposed expert-agnostic Intelligent Tutoring System performs effectively on the EdNet-KT4 dataset. Among all evaluated models, CNN and hybrid GRU-CNN achieved the highest accuracy (up to 98.41%), demonstrating strong generalization and stable learning behavior. Sequential and hybrid models consistently outperformed non-sequential approaches, highlighting the importance of capturing temporal learning patterns. The BiLSTM model also showed strong performance due to its bidirectional context modeling. Overall, the findings confirm that self-supervised deep learning can accurately predict student performance without expert annotations, making the system scalable and reliable for real-world educational platforms.

---

## Limitations & Future Work
Although the proposed system achieves high accuracy, it is currently evaluated only on interaction-based data from the EdNet-KT4 dataset, which may limit generalization across different educational platforms. The Transformer model also showed overfitting on shorter sequences, indicating the need for improved regularization and larger datasets.Future work will focus on incorporating multimodal inputs such as text, audio, and facial expressions, applying reinforcement learning for adaptive feedback, and improving model interpretability using explainable AI techniques. Expanding evaluation to low-resource environments and conducting user studies will further validate real-world applicability.

---

## Deployment Info
REST API (Flask): Load all models into memory, Receive JSON with flow features, Return predicted class + confidence.

---
