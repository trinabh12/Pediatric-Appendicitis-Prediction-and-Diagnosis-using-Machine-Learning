**Pipeline Overview**

This project implements a full-stack data engineering and machine learning pipeline for Pediatric Appendicitis Prediction using multimodal data, including clinical records, laboratory results, and ultrasound imaging.

The system is designed to be reproducible, versioned, and scalable, combining data ingestion, validation, feature engineering, lineage tracking, and multimodal model training into a unified workflow.

**Stage 1 — Data Ingestion**

The ingestion stage is responsible for acquiring raw data and organizing it into a structured format suitable for downstream processing. During this stage, dataset assets are downloaded from the source and metadata is extracted to understand schema, feature descriptions, and data types.

The outputs are organized into two primary directories:

image/ containing ultrasound scans

tabular/ containing clinical and laboratory records

This separation ensures modular processing of multimodal data while preserving traceability.

**Stage 2 — Data Validation and Profiling**

This stage ensures dataset integrity and establishes a data quality baseline before transformations are applied. Missing values are analyzed across all features and a missingness report is generated to guide imputation strategies.

Observed data is validated against metadata specifications to detect inconsistencies such as invalid categories, unexpected ranges, or schema mismatches. This prevents downstream failures and ensures feature assumptions remain valid.

**Stage 3 — Data Preparation**

The preparation stage cleans and structures the dataset for feature engineering and model training. Features are categorized based on value type, dependency relationships, and linkage to image assets.

Numeric features are imputed based on missingness levels and distribution skewness. Features with moderate missingness are filled using mean or median values, while binary and categorical features with low missingness are filled using mode. Dependent numeric features are recalculated after parent features are imputed to maintain consistency.

Logical dependencies are handled by distinguishing primary and secondary relationships. Missing values are filled using conditional indicators such as “Not observed” or “Not examined” to preserve semantic meaning.

For image data, ultrasound scans are parsed using their naming convention (<subject_id>.<view_id>.bmp). Images are grouped by view ID and an image registry is created, linking each subject to scan paths and sequence counts. This registry enables multimodal fusion in later stages.

**Stage 4 — Feature Engineering**

This stage focuses on deriving high-signal features from clinical, laboratory, and ultrasound data to improve predictive performance.

Clinical features capture symptom patterns and physical findings, while laboratory features quantify inflammatory markers and hematological indicators. Ultrasound features describe anatomical findings such as appendix diameter, tissue reactions, and fluid presence.

Derived features are created to encode clinically meaningful signals. Examples include:

Classic presentation flags combining symptom clusters

Inflammatory triage combining CRP and WBC levels

Left shift indicators reflecting neutrophil dominance

Pathological diameter flags for abnormal appendix size

Secondary findings scores aggregating ultrasound indicators

Image path features are generated from the image registry, linking each subject to view-specific scans. Categorical features are one-hot encoded and binary features are standardized to 0/1 to ensure compatibility with machine learning models.

**Stage 5 — Data Versioning and Lineage**

This stage introduces reproducibility and governance into the pipeline. A hash function is used to lock the processed dataset version, ensuring experiments can be traced back to a specific data state.

Lineage metadata records transformation steps and dataset provenance. The dataset is split into training (70%), validation (10%), and test (20%) sets, preserving reproducibility and preventing data leakage.

This stage enables reliable experimentation and supports auditability in research or production environments.

**Stage 6 — Model Training and Evaluation**

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
