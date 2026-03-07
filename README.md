# Reddit Mental Health Sentiment Analysis

## Overview

This project analyzes 3,553 Reddit posts from 10 mental health and crisis support communities, exploring how language, sentiment, and topics vary between communities. The goal of this is to understand not only *what* people discuss in these communities, but *how* the nature of that discussion varies. 

>**Content Warning:** This repository explores a dataset discussing trauma, domestic abuse, mental health crises, and financial hardship. Some posts may include distressing language.

## Dataset

This repository uses the [Dreaddit Dataset](https://arxiv.org/abs/1911.00133) (Turcan & McKeown, 2019). This dataset contains posts from ten subreddits spanning five domains: anxiety, PTSD, financial hardship, domestic abuse, and social stress. The combined test and train set contain 3,553 annotated Reddit posts. The dataset is not included in this repository, and must be downloaded from the [publication's repository](https://github.com/dreaddit/dreaddit).

Turcan, E., & McKeown, K. (2019, November). Dreaddit: A reddit dataset for stress analysis in social media. In Proceedings of the tenth international workshop on health text mining and information analysis (LOUHI 2019) (pp. 97-107).

## Research Questions
- **RQ1**: How do language and sentiment differ across communities?
- **RQ2**: What topics characterize each community?
- **RQ3**: How accurately can the community a post belongs to be predicted?
- **RQ4**: What features most distinguish between the different subreddits?

## Key Findings
1. **Post length and sentiment differ significantly across communities** (Kruskal-Wallis, p < 0.001). Crisis support communities (r/assistance, r/food_pantry) contained posts with more positive sentiment than trauma-focused communities (r/ptsd, r/domesticviolence), reflecting a potential difference between action-oriented help-seeking communities and those oriented toward processing traumatic experiences.
2. **Topic modeling revealed meaningful thematic variation across communities**. LDA identified distinct themes among the posts including, Financial Stress, Trauma Narratives, Generalized Anxiety, and Romantic & Social Relationships. Discussion of these topics were unevenly distributed across subreddits. Abuse-focused communities (r/survivorsofabuse and r/domesticviolence) scored highest on Anxiety & PTSD symptom management. Scores for these subreddits were higher than r/ptsd itself, suggesting these communities serve as spaces where trauma and its psychological consequences are discussed together. 
3. **Community of origin can be predicted with 46.2% accuracy** using language, sentiment, and topic features. This is more than double the 20.0% majority-class baseline. Furthermore, misclassification showed meaningful relationships. r/almosthomeless and r/food_pantry were most often confused with r/assistance, and r/ptsd was frequently misclassified as r/anxiety or r/stress, reinforcing thematic similarities between subreddits.
4. **Communities are more distinguished by *what* people discuss than *how* they write**. All 10 LDA topic features were consistent across all cross-validation folds, and ranked among the 20 features with the highest importance scores. All four sentiment features were also present, as well as community specific terms (e.g., "ptsd", "homeless", and "anxiety"). Notably, "tldr" (too long, didn't read) was a consistent predictor, suggesting some communities may be more oriented towards venting than concrete detail-oriented posts.


## Methodology
- **Text analysis**: TF-IDF word clouds and chi-squared test on word frequency distributions
- **Sentiment analysis**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Topic modeling**: Latent Dirichlet Allocation (LDA) with pyLDAvis visualization
- **Classification**: Random Forest with a 10-fold cross-validation; features include TF-IDF (500 terms), VADER sentiment scores, and LDA topic probabilities. TF-IDF and LDA features were recalculated every fold to prevent data leaks
- **Feature importance**: Random Forest impurity-based importance aggregated across all 10 folds; only features appearing in all 10 folds were analyzed to ensure validity in results.
- **Statistical testing**: Kruskal-Wallis and Chi-Squared tests

## Tools used
- Python
- NLTK
- Gensim
- pyLDAvis
- Scikit-learn
- Pandas
- NumPy
- SciPy 
- Plotly

## How to run
1. Clone the repo
2. Download the Dreaddit dataset, combine the test and training sets, and place it in a `dreaddit/` folder as a `dreaddit.csv` file.
3. Install dependencies: `pip install -r requirements.txt`
4. Run `sentiment-analysis.ipynb`