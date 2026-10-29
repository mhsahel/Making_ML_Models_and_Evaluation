# Chronic Kidney Disease (CKD) Prediction

Dataset description and project notes for a binary classification project on the
**Early stage of Indians Chronic Kidney Disease (CKD)** dataset from the UCI Machine Learning Repository.

> **Disclaimer:** This is a learning project. The model and findings are not for real medical use.

---

## 1. Dataset Overview

| Item | Detail |
|---|---|
| **Full title** | Early stage of Indians Chronic Kidney Disease (CKD) |
| **Repository** | UCI Machine Learning Repository |
| **Source** | Dr. P. Soundarapandian (nephrologist), Apollo Hospitals, Karaikudi, Tamil Nadu, India |
| **Creator** | L. Jerlin Rubini, Alagappa University |
| **Date collected** | July 2015 |
| **Task** | Binary classification: does the patient have CKD (`ckd`) or not (`notckd`)? |
| **Size** | 400 patients, 24 features + 1 target |
| **Class balance** | 250 `ckd` (62.5%), 150 `notckd` (37.5%) |
| **Missing values** | Yes, written as `?` (about 10% of all cells) |

Each row is one patient. The features come from three kinds of information:

- **Urine tests:** specific gravity, albumin, sugar, red blood cells, pus cells, pus cell clumps, bacteria
- **Blood tests:** glucose, urea, creatinine, sodium, potassium, hemoglobin, packed cell volume, white and red blood cell counts
- **Medical history and symptoms:** hypertension, diabetes, coronary artery disease, appetite, pedal edema, anemia

---

## 2. Feature Dictionary

### Numeric features (11)

| Column | Meaning | Unit |
|---|---|---|
| `age` | Age | years |
| `bp` | Blood pressure | mm/Hg |
| `bgr` | Blood glucose (random) | mg/dl |
| `bu` | Blood urea | mg/dl |
| `sc` | Serum creatinine | mg/dl |
| `sod` | Sodium | mEq/L |
| `pot` | Potassium | mEq/L |
| `hemo` | Hemoglobin | gms |
| `pcv` | Packed cell volume | not stated |
| `wc` (`wbcc` in the ARFF file) | White blood cell count | cells/cumm |
| `rc` (`rbcc` in the ARFF file) | Red blood cell count | millions/cmm |

### Categorical features (14, including the target)

| Column | Meaning | Values |
|---|---|---|
| `sg` | Specific gravity | 1.005, 1.010, 1.015, 1.020, 1.025 |
| `al` | Albumin | 0 to 5 |
| `su` | Sugar | 0 to 5 |
| `rbc` | Red blood cells | normal / abnormal |
| `pc` | Pus cell | normal / abnormal |
| `pcc` | Pus cell clumps | present / notpresent |
| `ba` | Bacteria | present / notpresent |
| `htn` | Hypertension | yes / no |
| `dm` | Diabetes mellitus | yes / no |
| `cad` | Coronary artery disease | yes / no |
| `appet` | Appetite | good / poor |
| `pe` | Pedal edema | yes / no |
| `ane` | Anemia | yes / no |
| `class` | **Target** | ckd / notckd |

---

## 3. Data Quality Issues

- **Missing values:** 243 of 400 rows have at least one missing value. The worst columns are
  `rbc` (152 missing), `rbcc` (131), `wbcc` (106), `pot` (88), `sod` (87) and `pcv` (71).
- **Formatting errors in the file:** a few rows have stray commas or an empty field that shifts the columns.
  They must be fixed before loading.
- **Column name mismatch:** the info file says `wc` and `rc`, but the ARFF file uses `wbcc` and `rbcc`.
- **Missing unit:** the info file gives no unit for `pcv`.
- **Class count:** a naive parse gives 149 `notckd` plus one broken row. After fixing the file, the count is 150.
- **Ordinal columns:** `sg`, `al` and `su` are described as nominal, but their values have a natural order.

---

## 4. Loading the Data

The data ships as `Chronic_Kidney_Disease.rar`. Extract it first (7-Zip, WinRAR, or `unrar` on Colab/Linux),
then load `chronic_kidney_disease_full.arff` with a small parser that fixes the broken rows.

```python
import io
import pandas as pd

path = "Chronic_Kidney_Disease/chronic_kidney_disease_full.arff"

with open(path, encoding="utf-8", errors="ignore") as f:
    lines = [l.strip() for l in f.read().splitlines()]

# Column names from the @attribute lines
cols = [l.split("'")[1] for l in lines if l.lower().startswith("@attribute")]

# Data rows (everything after @data)
start = [i for i, l in enumerate(lines) if l.lower() == "@data"][0]
rows = [l for l in lines[start + 1:] if l and not l.startswith("%")]

# Fix broken rows: strip spaces and drop empty fields caused by stray commas
fixed = []
for r in rows:
    parts = [p.strip() for p in r.split(",")]
    if len(parts) > len(cols):
        parts = [p for p in parts if p != ""]
    fixed.append(",".join(parts))

df = pd.read_csv(io.StringIO("\n".join(fixed)), names=cols, na_values="?")

print(df.shape)                    # (400, 25)
print(df["class"].value_counts())  # ckd 250, notckd 150
```

Useful next steps:

```python
df["class"] = df["class"].map({"ckd": 1, "notckd": 0})   # encode the target
df.isna().sum().sort_values(ascending=False)             # missing values per column
```

Do the train/test split **before** imputing missing values, so the test set does not leak into the imputer.

---

## 5. Limitations

- **Small dataset:** 400 rows means results are unstable. Use cross-validation, not a single train/test split.
- **One hospital, one region, one year:** the model may not generalize to patients elsewhere.
- **Easy classes:** several features separate the classes very strongly, so many models reach 97-99% accuracy.
  A high score alone does not prove a good model.
- **Unclear definitions:** the dataset documentation does not explain how `notckd` patients were chosen or
  how CKD was diagnosed, so avoid strong medical claims.

---

## 6. Project Questions

1. Which imputation method works best: median/mode, KNN, or iterative imputation?
2. Which model gives the best **recall** for `ckd`? Missing a sick patient is worse than a false alarm.
3. Which features matter most, and does this agree with medical knowledge (for example creatinine, hemoglobin, albumin)?
4. How well does a model perform with fewer, cheaper tests?

---

## 7. Planned Workflow

- [ ] Extract and load the data, fix formatting errors
- [ ] Exploratory data analysis (distributions, missingness, correlations)
- [ ] Preprocessing: encode categories, impute missing values (compare methods)
- [ ] Train and compare models (logistic regression, random forest, gradient boosting, etc.)
- [ ] Evaluate with cross-validation (precision, recall, F1, confusion matrix)
- [ ] Feature importance and interpretation
- [ ] Write up results and limitations

---

## 8. Credits

Dataset: *Early stage of Indians Chronic Kidney Disease (CKD)*, collected by L. Jerlin Rubini under the
guidance of Dr. P. Eswaran (Alagappa University), with data from Dr. P. Soundarapandian, Apollo Hospitals,
Karaikudi, Tamil Nadu, India. Distributed through the UCI Machine Learning Repository.

