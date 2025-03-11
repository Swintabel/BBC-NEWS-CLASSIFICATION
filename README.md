# BBC-NEWS-CLASSIFICATION

### PROJECT SUMMARY
This project classifies BBC News articles into five categories using machine learning and deep learning methods. It covers data preprocessing, feature extraction, model selection, and neural network implementation. The model can be applied to personalize news delivery, improve content discovery, and enhance content management for real-time systems and platforms.

### **Key Technologies**

**Python:** Used as the primary programming language for implementing the entire text classification pipeline.

**Scikit-learn:** Utilized for implementing machine learning models (SVC, AdaBoost, etc.), feature extraction (TF-IDF), and model evaluation.

**TensorFlow:** Used to implement and train the Convolutional Neural Network (CNN) for the text classification task.

**Numpy:** Employed for numerical operations, such as handling arrays and performing matrix computations in model training.

**Pandas:** Used for data loading, cleaning, and preprocessing the dataset, including text normalization.

**Matplotlib:** Used for visualizing model performance metrics and training progress.

**SMOTE:** Applied to balance the class distribution in the dataset by generating synthetic minority class examples.

**PCA:** Applied for dimensionality reduction to improve model performance and reduce overfitting.

### Skills Highlighted

- Text Preprocessing
- Feature Extraction (BoW, N-grams, TF-IDF)
- Model Selection and Evaluation
- Machine Learning Algorithms (SVC, AdaBoost, Logistic Regression)
- Neural Networks (CNN)
- Dimensionality Reduction (PCA)
- Data Augmentation (SMOTE)
- Model Training and Tuning

### The process

---

**Step 1: Data Loading & Description**
The dataset consists of 2225 news articles, categorized into five classes: 
Business: 510 articles
Entertainment: 386 articles
Politics: 417 articles
Sports: 511 articles
Tech: 401 articles
We preprocess these articles to prepare them for machine learning, with special attention to the imbalanced data problem, where certain categories have more articles than others.
Key Codes: `pd.read_csv`, `os.walk`

**Step 2: Data Preprocessing**
Preprocessing text data is crucial to make it usable for machine learning models. This involves:
**Text Normalization:** Removing special characters, extra spaces, and converting all text to lowercase to standardize the data.
**Tokenization:** Splitting text into individual words or tokens to make it easier to process.
Stopword Removal: Eliminating common words (e.g., "the", "a") that do not contribute meaningful information to the analysis.
**Lemmatization:** Reducing words to their root forms (e.g., “running” to “run”) to ensure consistency and improve model performance.
Key Codes:  `nltk.tokenize`, `nltk.corpus.stopwords`

**Step 3: Feature Extraction**
Once the text is cleaned, it needs to be converted into a numerical format that machine learning algorithms can process. We use three methods for feature extraction:
**Bag of Words (BoW):** A simple technique that counts the frequency of each word in the text. It does not account for word order or meaning.
**N-grams:** Builds on BoW by considering sequences of N words, capturing some grammatical context but can lead to large, sparse matrices.
**TF-IDF:** Weighs words based on their frequency in the document and how rare they are across the entire dataset. This helps prioritize unique and important words.
Key Codes: `TfidfVectorizer`, `CountVectorizer`, `n-grams`

**Step 4: Model Selection**
We evaluate several machine learning models to find the most suitable classifier for this task. The models considered include: AdaBoost, Gradient Boosting, Support Vector Classifier (SVC), Logistic Regression,K-Nearest Neighbors (KNN). We apply Principal Component Analysis (PCA) to reduce dimensionality and enhance model performance.
After testing all models, we choose SVC with TF-IDF features and 15 PCA components, as it delivers the best performance.
Key Codes: `SVC`, `PCA`, `fit_transform`

**Step 5: Model Evaluation**
The selected model is evaluated on both the training set and test set to gauge its performance. We focus on metrics such as:
**Accuracy:** Percentage of correct predictions.
**Precision:** Ratio of true positives to the total predicted positives.
**Confusion Matrix:** Helps us understand false positives/negatives and true positives/negatives.
These results indicate the model generalizes well and is not overfitting to the training data.
Key Codes: `accuracy_score`, `confusion_matrix`

**Step 6: Neural Network Approach (CNN)**
To explore a deep learning approach, we implement a Convolutional Neural Network (CNN) for text classification. The steps include:
**Data Preparation:** Tokenizing the text, padding sequences to ensure uniform input length, and handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
**Model Architecture:**
Embedding layer to convert words into dense vectors.
Convolutional layers with ReLU activation.
Max pooling for dimensionality reduction.
Fully connected layers with dropout to prevent overfitting.
**Training:** The model is trained with 20 epochs, using TensorBoard for logging, ModelCheckpoint to save the best model, and ReduceLROnPlateau to adjust the learning rate when accuracy plateaus.
Key Codes: `SMOTE`, `fit`, `Sequential`, `Conv1D`

### Key Results

---

- The **SVC model demonstrated superior performance** in terms of accuracy, compared to other models tested.

The model achieves:
Training Accuracy: 95%
Test Accuracy: 96.7%

- The CNN model showed promising results, though it did not surpass the SVC model in accuracy. Further optimization could improve its performance.

Training Accuracy: 92%
Test Accuracy: 94%
