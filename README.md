# evt_w1_gain

Graduate course work for **Missing Data Imputation Methods**  
Selçuk University, Computer Engineering MSc Program

## Topic

**Missing Data Imputation with Generative Adversarial Networks (GAIN)**  
Adversarial Generative Missing Data Imputation with GAIN

### Primary Paper

Yoon, J., Jordon, J., & van der Schaar, M. (2018).  
*GAIN: Missing Data Imputation using Generative Adversarial Nets.*  
Proceedings of the 35th International Conference on Machine Learning.

## Scope

The project covers:

- Missing data fundamentals
- Generative Adversarial Networks
- GAIN architecture
- Generator and discriminator
- Observation mask
- Hint mechanism
- Adversarial and reconstruction losses
- Training and inference procedures
- Strengths and limitations of GAIN
- Experimental comparison with KNN and MICE

## Planned Experiment

- Dataset: UCI Spambase
- Missingness mechanism: MCAR
- Missing rate: 20%
- Methods:
  - GAIN
  - KNN Imputation
  - MICE / Iterative Imputation
- Same missingness mask for all methods
- Primary metric: RMSE on artificially masked values
- Additional evaluation: runtime and repeated runs with different random seeds

## Repository Structure

- `00_guidelines/` - Course guidelines
- `01_sources/` - Academic papers and references
- `02_reading_notes/` - Reading notes
- `03_experiments/` - Experimental code, data and results
- `04_figures_tables/` - Figures and tables
- `05_report/` - Report drafts
- `06_video/` - Video-related files

## Deadline

Official deadline: **20 October 2026**

Internal target: **3 October 2026**
