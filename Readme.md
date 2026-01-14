## ▶️ How to Run the DDx System

Follow the steps below to run the knowledge-graph–based differential diagnosis pipeline.

---

### 1️ Download the Dataset

This project uses the **DDXPlus Dataset (English)**, which is not included in the repository due to its size.

Please follow the instructions in:

```Data/Readme.md```

Download the dataset and extract it into:

```Data/ddxplus```

---

### 2 Prerequisites: Install Dependencies

Install the required Python packages by running:

```pip install -r requirements.txt```

Ensure all dependencies are installed before proceeding with the dataset download.

---

### 3 Build the Knowledge Graph

Once the dataset is available, construct the knowledge graph by running the notebook:

```KG_Construction```

This notebook processes the dataset and generates the serialized knowledge graph:

```kg.pkl```

Ensure that `kg.pkl` is saved in the expected location used by the runtime scripts.

---

### 4 Run the Differential Diagnosis Engine

After the knowledge graph is built, run the interactive DDx system using:

```python -m scripts.run_ddx```

You will be prompted to:

- Enter free-text symptom descriptions
- Answer follow-up diagnostic questions interactively

The system outputs a ranked list of candidate conditions.


