# Dataset source and attribution

This project uses selected academic columns from the **Student Performance** dataset published by the UCI Machine Learning Repository. It uses the Portuguese-language course file (649 records), retaining `studytime`, `absences`, `G1`, `G2` and `G3`.

- Official dataset: https://archive.ics.uci.edu/dataset/320/student+performance
- DOI: https://doi.org/10.24432/C5TG7T
- Original authors: Paulo Cortez and Alice Silva (2008).
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated by UCI.
- Original file: `student-por.csv`; the project CSV is a five-column working extract.

| Field | Meaning |
|---|---|
| `studytime` | Weekly study-time band: 1 = less than 2 hours; 2 = 2-5 hours; 3 = 5-10 hours; 4 = more than 10 hours. |
| `absences` | Count of school absences; it is not an attendance percentage. |
| `G1` | First-period grade. |
| `G2` | Second-period grade. |
| `G3` | Final-period grade and prediction target, on a 0-20 scale. |

The official dataset covers student achievement at two Portuguese schools. The original full dataset has 649 records and 33 columns. A check of the included file found 0 exact duplicate data rows and 0 blank cells. Earlier-period grades G1 and G2 are strongly correlated with G3. Accordingly, the model is only suitable once those earlier grades are available; it is not an early-semester prediction model.

The dataset does not contain exact study hours or attendance percentages. No such values are fabricated or inferred.
