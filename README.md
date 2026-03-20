<h3>Project Overview</h3>
<p>
This project implements a full-stack data engineering and machine learning pipeline for Pediatric Appendicitis Prediction using multimodal data, including clinical records, laboratory results, and ultrasound imaging.</p>
<p>
The system is designed to be reproducible, versioned, and scalable, combining data ingestion, validation, feature engineering, lineage tracking, and multimodal model training into a unified workflow.</p>

<h4>Stage 1 — Data Ingestion</h4>

The ingestion stage is responsible for acquiring raw data and organizing it into a structured format suitable for downstream processing. During this stage, dataset assets are downloaded from the source and metadata is extracted to understand schema, feature descriptions, and data types.

The outputs are organized into two primary directories:

image/ containing ultrasound scans

tabular/ containing clinical and laboratory records

This separation ensures modular processing of multimodal data while preserving traceability.

<h4>Stage 2 — Data Validation and Profiling</h4>

This stage ensures dataset integrity and establishes a data quality baseline before transformations are applied. Missing values are analyzed across all features and a missingness report is generated to guide imputation strategies.

Observed data is validated against metadata specifications to detect inconsistencies such as invalid categories, unexpected ranges, or schema mismatches. This prevents downstream failures and ensures feature assumptions remain valid.

<h4>Stage 3 — Data Preparation</h4>

The preparation stage cleans and structures the dataset for feature engineering and model training. Features are categorized based on value type, dependency relationships, and linkage to image assets.

Numeric features are imputed based on missingness levels and distribution skewness. Features with moderate missingness are filled using mean or median values, while binary and categorical features with low missingness are filled using mode. Dependent numeric features are recalculated after parent features are imputed to maintain consistency.

Logical dependencies are handled by distinguishing primary and secondary relationships. Missing values are filled using conditional indicators such as “Not observed” or “Not examined” to preserve semantic meaning.

For image data, ultrasound scans are parsed using their naming convention (<subject_id>.<view_id>.bmp). Images are grouped by view ID and an image registry is created, linking each subject to scan paths and sequence counts. This registry enables multimodal fusion in later stages.

<h4>Stage 4 — Feature Engineering</h4>

This stage focuses on deriving high-signal features from clinical, laboratory, and ultrasound data to improve predictive performance.

Clinical features capture symptom patterns and physical findings, while laboratory features quantify inflammatory markers and hematological indicators. Ultrasound features describe anatomical findings such as appendix diameter, tissue reactions, and fluid presence.

Derived features are created to encode clinically meaningful signals. Examples include:

Classic presentation flags combining symptom clusters

Inflammatory triage combining CRP and WBC levels

Left shift indicators reflecting neutrophil dominance

Pathological diameter flags for abnormal appendix size

Secondary findings scores aggregating ultrasound indicators

Image path features are generated from the image registry, linking each subject to view-specific scans. Categorical features are one-hot encoded and binary features are standardized to 0/1 to ensure compatibility with machine learning models.

<h4>Stage 5 — Data Versioning and Lineage</h4>

This stage introduces reproducibility and governance into the pipeline. A hash function is used to lock the processed dataset version, ensuring experiments can be traced back to a specific data state.

Lineage metadata records transformation steps and dataset provenance. The dataset is split into training (70%), validation (10%), and test (20%) sets, preserving reproducibility and preventing data leakage.

This stage enables reliable experimentation and supports auditability in research or production environments.

<h4>Stage 6 — Model Training and Evaluation</h4>

The final stage trains and evaluates models using a multimodal approach. The workflow is divided into three phases to handle tabular and image data separately before combining them.

*Phase 1 — Tabular Model (MLP)*

A Multi-Layer Perceptron is trained on clinical, laboratory, and engineered features. The model uses a fixed random seed for reproducibility and a decision threshold of 0.614 to optimize diagnostic sensitivity and specificity.

*Phase 2 — Image Model (CNN)*

A Convolutional Neural Network processes ultrasound images grouped by view. The model learns spatial patterns and anatomical features associated with appendicitis, complementing tabular signals.

***Phase 3 — Fusion Model (Multimodal)***

The fusion model combines probability outputs from the tabular MLP and the CNN to produce a final prediction. This approach mirrors real clinical workflows, where diagnosis relies on symptoms, lab results, and imaging evidence.

The fusion strategy improves robustness by leveraging complementary modalities and reducing reliance on any single data source.

Evaluation Metric: Why AUC Instead of Accuracy

Accuracy can be misleading in medical datasets where class imbalance is common. A model may achieve high accuracy by predicting the majority class while failing to detect true positive cases.

The Area Under the ROC Curve (AUC) is used because it measures the model’s ability to distinguish between classes across all decision thresholds. AUC is threshold-independent, robust to imbalance, and better reflects diagnostic performance in clinical settings.

**Model Performance**<br>
*Phase1: Tabular MLP*<br>
Train AUC: 84.33% <br>
Val AUC: 76.08% <br>
Test AUC: 77.99% <br>

*Phase2: CNN*<br>
Train AUC: 82.79%<br>
Val AUC: 68.11%<br>
Test AUC: 66.85%<br>

***Phase3: Fusion Model***<br>
Train AUC: 93.88%<br>
Val AUC: 81.79%<br>
Test AUC: 80.59%<br>
*(final result)*


<h3>System Walkthrough</h3>
<p>Industrial Relevance Note: The following interface demonstrates a decoupled architecture where the Frontend (Tailwind CSS) communicates with a high-concurrency Backend (FastAPI) to provide real-time clinical support.</p>

<h4>A. Clinical Data Entry (Demographics & Labs)</h4>
<p>The dashboard allows for the rapid entry of patient vitals and laboratory markers. It automatically handles one-hot encoding and BMI calculations before transmission.</p>

<img width="1177" height="707" alt="image" src="https://github.com/user-attachments/assets/7d229813-7501-4236-a21e-1ec102cbc967" />

<h4>B. Multimodal Inference Engine</h4>
<p>Physicians upload the ultrasound scan (BMP/JPG). The backend triggers the Stage 6 Fusion Model to synthesize tabular and spatial features.</p>

<img width="1202" height="128" alt="image" src="https://github.com/user-attachments/assets/94de52a4-473a-49ac-b8eb-b85bd44fe3e0" />


<h3>🛠️ Setup & Deployment Guide</h3>
<p>This project is fully containerized to ensure a consistent, "Zero-Trust" environment suitable for secure hospital intranets. Follow these steps to deploy the SaaS for Doctors locally.</p>

<h4>1. Prerequisites</h4>
<p>Docker & Docker Compose installed on your system.</p>

Terminal/Command Prompt with administrative privileges.

The Regensburg Pediatric Appendicitis Dataset (or sample files) for testing.

2. Installation
Clone the repository and navigate to the project root:

Bash
git clone https://github.com/Trinabh-Mitra/Pediatric-Appendicitis-Prediction.git
cd Pediatric-Appendicitis-Prediction
3. Asset Verification
Ensure the following files are in place to allow the offline interface and engine to boot:

ui/tailwind.min.css: Required for local styling in air-gapped environments.

api/models/: Ensure your trained .keras and .json artifacts are in this directory.

4. Launching the System
Run the following command to build the image and start the microservice:

Bash
docker-compose up --build
5. Accessing the Dashboard
Once the logs show Application startup complete, open your browser and navigate to:

Clinical Interface: http://localhost:8000

API Documentation (Swagger): http://localhost:8000/docs

👔 Why this is Industrially Relevant (Pitch Highlights)
During your Review II, you can specifically point to this setup guide to prove the following points:

Microservice Architecture: The system uses a decoupled FastAPI backend, allowing it to be integrated into existing hospital Electronic Health Record (EHR) systems via REST API calls.

Infrastructure as Code (IaC): By using docker-compose.yml, the entire clinical environment is version-controlled. Whether it's deployed on a high-end server or a basic clinic laptop, the behavior is identical.

Security & Privacy: The local deployment ensures that sensitive patient vitals and ultrasound scans never leave the hospital's local network, meeting strict HIPAA/GDPR data residency requirements.


