# CAN-Ranker

> AI-Powered Candidate Discovery, Ranking & Evaluation System

## Overview

CAN-Ranker is an intelligent candidate ranking system designed to evaluate large candidate datasets against a given Job Description (JD) and identify the best-fit candidates through explainable, multi-factor scoring.

The system combines structured feature extraction, semantic matching, behavioral analysis, profile consistency validation, and weighted ranking to generate accurate and explainable candidate recommendations.

Designed with scalability in mind, CAN-Ranker efficiently processes large candidate datasets while remaining modular, configurable, and production-ready.

---

## Features

- Intelligent Job Description Analysis
- Large-Scale Candidate Processing
- Semantic Candidate Matching
- Feature Engineering Pipeline
- Candidate Consistency Validation
- Behavioral Signal Analysis
- Suspicious Profile Detection
- Explainable AI Ranking
- Multi-Factor Weighted Scoring
- Automatic Top-N Candidate Selection
- Competition & Production Ready CSV Export

---

# Architecture

```
                    Job Description
                           │
                           ▼
                  JD Feature Extraction
                           │
                           ▼
                  Candidate Loader
                           │
                           ▼
                  Feature Extraction
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
 Career Analysis     Skill Analysis     Signal Analysis
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
            Profile Consistency Validation
                           │
                           ▼
              Candidate Matching Engine
                           │
                           ▼
                Multi-Factor Scoring Engine
                           │
                           ▼
                 Top Candidate Selection
                           │
                           ▼
                Reasoning Generation Engine
                           │
                           ▼
                   Submission CSV Export
```

---

# Repository Structure

```
can-ranker/
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
│
├── docs/
│
├── logs/
│
├── src/
│   ├── parser/
│   ├── features/
│   ├── consistency/
│   ├── fraud/
│   ├── matcher/
│   ├── scoring/
│   ├── reasoning/
│   ├── exporter/
│   └── utils/
│
├── tests/
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Core Modules

### Candidate Parser

Efficiently loads and processes candidate data from large JSONL datasets.

### Feature Engineering

Extracts meaningful technical and professional attributes including:

- Skills
- Experience
- Career History
- Education
- Certifications
- Projects
- Behavioral Signals

### Consistency Engine

Evaluates profile consistency across multiple sections to identify conflicting or weakly supported information.

### Matching Engine

Measures candidate alignment with the target Job Description using semantic and structured analysis.

### Scoring Engine

Calculates a normalized ranking score by combining multiple evaluation factors into a single candidate score.

### Reasoning Engine

Generates concise, evidence-based explanations describing why a candidate received a particular ranking.

### Export Engine

Produces submission-ready CSV files compatible with downstream evaluation systems.

---

# Processing Pipeline

```
Load Job Description
        │
        ▼
Load Candidate Profiles
        │
        ▼
Extract Candidate Features
        │
        ▼
Validate Profile Consistency
        │
        ▼
Analyze Behavioral Signals
        │
        ▼
Calculate Candidate–JD Match
        │
        ▼
Generate Final Ranking Score
        │
        ▼
Select Top Candidates
        │
        ▼
Generate Candidate Reasoning
        │
        ▼
Export CSV
```

---

# Input Data

Place all input files inside:

```
data/raw/
```

Example:

```
candidates.jsonl.gz
candidate_schema.json
job_description.md
submission_spec.md
sample_candidates.json
sample_submission.csv
redrob_signals_doc.md
```

Large datasets are intentionally excluded from version control.

---

# Output

The generated submission is stored in:

```
data/output/
```

Example format:

```csv
candidate_id,rank,score,reasoning
```

Example:

```csv
candidate_id,rank,score,reasoning
CAND_0000123,1,0.987,"Strong production AI engineering experience with retrieval systems, Python, and vector databases. Closely aligns with the target role while demonstrating consistent professional progression."
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/0xgaurav/can-ranker.git

cd can-ranker
```

Create a virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run

```bash
python main.py
```

---

# Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Sentence Transformers
- PyYAML

---

# Design Principles

- Modular Architecture
- Explainable Rankings
- Configurable Scoring
- Efficient Large-Scale Processing
- Production-Oriented Design
- Maintainable Codebase
- CPU-Friendly Execution

---

# Future Improvements

- Learning-to-Rank models
- Feedback-driven ranking optimization
- Candidate clustering
- Resume parsing support
- Recruiter feedback integration
- Interactive ranking dashboard
- Advanced anomaly detection

---

# License

This project is released under the MIT License.

---

## Author

**Gaurav Singh**

Computer Science Engineering (AI) Student

Passionate about Artificial Intelligence, Machine Learning, Search Systems, and Intelligent Ranking Algorithms.
