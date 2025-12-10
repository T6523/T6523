# Krittitee Naulkhao

**Computer Science Student (B.E.) | Data Science & Machine Learning**

I am a Computer Science undergraduate at Chulalongkorn University (GPA 4.00) with a focus on Applied Machine Learning, Computer Vision, and Mathematical Optimization.

### ⚠️ Note on Public Repositories

Most of my recent development work involves proprietary datasets (Medical Imaging, Financial Risk Data) and confidential research protected by Non-Disclosure Agreements (NDAs). Consequently, I cannot make the source code for these projects public.

Below is a summary of the architectures, technical stacks, and methodologies I have implemented in these private repositories.

---

### Technical Skills

* **Languages:** Python, SQL, Java
* **Machine Learning:** PyTorch, Tensorflow, Scikit-Learn, XGBoost, CatBoost, LightGBM, ModernBERT, YOLO, Faster R-CNN
* **Data Engineering:** Pandas, NumPy, Albumentations, FAISS (Vector DB)
* **Deployment & Tools:** Streamlit, Git, Linux (WSL)

---

### Key Projects (Private/NDA)

#### 1. Credit Risk Prediction Engine (AI Hack 2025)
* **Context:** Developed a financial risk model for a high-dimensional dataset (40+ columns, 32,000 rows).
* **Stack:** Python, CatBoost, XGBoost, LightGBM, AutoGluon, TabNet.
* **Implementation Details:**
    * Engineered a data cleaning pipeline handling sparse categorical classes and outlier clipping (2SD).
    * Synthesized financial indicators through domain-specific feature engineering.
    * Designed a stacked ensemble architecture comparing tree-based gradient boosting methods against deep learning approaches (TabNet).

#### 2. Medical Imaging Baseline Models (I-Square Hackathon)
* **Context:** Served as Technical Lead to develop the official competition benchmarks for Google & Gulf-sponsored tracks.
* **Stack:** ResNet50, YOLO, Faster R-CNN, Albumentations.
* **Implementation Details:**
    * **Classification Track:** Implemented ResNet50 with grayscale and denoising preprocessing steps. Solved severe class imbalance issues using synthetic oversampling and Albumentations.
    * **Segmentation Track:** Built an end-to-end object detection pipeline using YOLO and Faster R-CNN architectures.

#### 3. Academic Collaboration Search (NLP & RAG System)
* **Context:** University final project utilizing proprietary Scopus publication data to recommend research partners.
* **Stack:** ModernBERT, DistilRoBERTa, FAISS, Streamlit.
* **Implementation Details:**
    * **Classification:** Fine-tuned ModernBERT on research paper titles and abstracts to categorize academic subjects.
    * **Retrieval-Augmented Generation (RAG):** Generated embeddings using DistilRoBERTa and indexed them in a FAISS vector database to enable millisecond-latency nearest neighbor search.
    * **Deployment:** Visualized prediction confidence and collaborative statistics via a Streamlit dashboard.

---

### Contact
* **LinkedIn:** https://www.linkedin.com/in/krittiteenaulkhao/
* **Email:** krittitee.n07@gmail.com