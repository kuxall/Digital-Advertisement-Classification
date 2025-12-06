NAME: Aashiq Chalise	STUDENT NUMBER: 250214460
PROPOSED TITLE OF PROJECT: Digital Advertisement Classification System using ML
BRIEFLY DESCRIBE YOUR FIELD OF STUDY:
Computer Science with a specialization in Artificial Intelligence and Machine Learning. The specific focus is on Natural Language Processing (NLP) for automated text classification. This project explores the application of supervised learning algorithms, particularly Support Vector Machines (LinearSVC) and Term Frequency-Inverse Document Frequency (TF-IDF) vectorization, to solve unstructured data categorization problems in digital advertising. The study bridges theoretical machine learning concepts with practical software engineering by deploying the model as an end-user SaaS application.



WHAT QUESTION DOES YOUR PROJECT SEEK TO ANSWER? 
The primary research question is: To what extent can a machine learning-based system utilizing Linear Support Vector Classifiers (LinearSVC) automate the categorization of unstructured digital advertisement text into specific industrial domains (e.g., 'Jobs – Retail', 'Sell – House', 'Banking') with enterprise-grade accuracy (>95%)?

Sub-questions include:
1. How does the TF-IDF feature extraction technique impact the model's ability to distinguish between semantically similar advertisement categories?
2. Can a confident-thresholding mechanism (e.g., probability < 60%) effectively mitigate false positives in real-world deployment scenarios compared to standard forced-choice classification?
3. profound improvements in processing speed and scalability can be achieved when replacing manual human classification with an automated Batch Processing pipeline?



WHAT HYPOTHESIS ARE YOU SEEKING TO TEST?
It is hypothesized that a Supervised Machine Learning model, trained on specific keyword domains involved in digital advertising, can achieve a classification accuracy exceeding 95%, thereby outperforming traditional rule-based systems or manual tagging in terms of speed and consistency.

Key Hypotheses:
1. **Accuracy**: A LinearSVC model optimized with TF-IDF vectorization will achieve >90% accuracy on the test dataset.
2. **Efficiency**: The automated system will reduce the time required to classify a batch of 1,000 ads from hours (manual) to seconds (automated).
3. **Reliability**: Implementing a confidence threshold mechanism (rejecting predictions <60%) will significantly reduce critical errors, making the system viable for unsupervised enterprise use.





WHAT ARE THE PROBABLE PROJECT OUTCOMES?
The project will deliver a fully integrated software solution, "AdClassifier Pro," comprising the following tangible outcomes:

1. **High-Performance ML Model**:
   - A serialized `LinearSVC` model trained on the `ConcatenatedDigitalAdData.xlsx` dataset.
   - Capability to classify text into major categories such as 'Jobs – Retail', 'Sell – House', 'Banking', 'Rent – Apartment', and 'Jobs – IT'.
   - Validated accuracy metrics (Target: ~96.5%) and confusion matrices.

2. **Streamlit Web Application**:
   - A user-friendly, responsive web interface built with Python/Streamlit.
   - **Real-time Classifier**: A text input interface for instant single-ad categorization with visual probability bars.
   - **Batch Processor**: A drag-and-drop tool for uploading CSV/Excel files to classify thousands of records simultaneously, including auto-generated downloadable reports.

3. **Analytics Dashboard**:
   - An "Executive Dashboard" module providing real-time insights into data distribution.
   - Interactive visualizations showing the frequency of advertisement types and model confidence metrics.

4. **Robust Documentation & Codebase**:
   - A comprehensive GitHub repository with clean, modular code (`streamlit_app.py`, `train_model.py`).
   - Technical documentation including a `README.md` with setup instructions and extensive code comments explaining the NLP pipeline.






PLEASE PROVIDE A BRIEF BIBLIOGRPAHY OF 2-4 KEY TEXTS FOR YOUR STUDY (USE HARVARD REFERENCE STYLE)
1. Jurafsky, D. and Martin, J.H. (2024) *Speech and Language Processing*. 3rd edn. Draft. Available at: https://web.stanford.edu/~jurafsky/slp3/ (Accessed: 5 December 2025).
2. Müller, A.C. and Guido, S. (2016) *Introduction to Machine Learning with Python: A Guide for Data Scientists*. Sebastopol, CA: O'Reilly Media.
3. Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.
4. Streamlit Inc. (2025) *Streamlit Documentation*. Available at: https://docs.streamlit.io (Accessed: 5 December 2025).






 
PLEASE NAME ANY MEMBER OF THE ACADEMIC TEAM YOU HAVE DISCUSSED THIS POTENTIAL PROJECT:



(staff use only) Project Approved by Academic Team?	YES		NO	
Any other Academic Staff comments
