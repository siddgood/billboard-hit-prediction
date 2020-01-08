# Billboard Hot 100 Hit Prediction
:notes: Predicting Billboard's Year-End Hot 100 Songs using audio features from Spotify and lyrics from Musixmatch

## Overview
Each year, Billboard publishes its Year-End Hot 100 songs list, which denotes the top 100 songs of that year. The objective of this project was to see whether or not a machine learning classifier could predict whether a song would become a hit *(known as [Hit Song Science](https://en.wikipedia.org/wiki/Hit_Song_Science))* given its intrinsic audio features as well as lyrics.

## Data and Features
A sample of 19000 Spotify songs was downloaded from [Kaggle](https://www.kaggle.com/edalrami/19000-spotify-songs), which included songs from various Spotify albums. Additionally, Billboard charts from 1964-2018 were scraped from Billboard and Wikipedia.

Using Spotify's Audio Features & Analysis API, the following [features](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/) were collected for each song: 
- **Mood**: Danceability, Valence, Energy, Tempo
- **Properties**: Loudness, Speechiness, Instrumentalness
- **Context**: Liveness, Acousticness

Additonally, lyrics were collected for each song using the [Musixmatch API](https://developer.musixmatch.com/documentation/api-reference/track-lyrics-get). I took a bag-of-words NLP approach to build a highly sparse (86%) matrix of unique words.

After cleaning the data, a dataset of approx. 10000 songs was created.

![](images/data-distribution.png)
![](images/fig-vs-decade.png)
![](images/genre-dist.png)

## Exploratory Data Analysis

**Spotify Features over Time**
![](images/acoustic-vs-time.png)
![](images/dance-vs-time.png)
![](images/energy-vs-time.png)
![](images/live-vs-time.png)
![](images/loud-vs-time.png)
![](images/speech-vs-time.png)

The above graphs clearly show that audio features evolve over time. More importantly, the separability of data in certain graphs such as *Acousticness vs. Time* and *Loudness vs. Time* indicates potentially significant features that can help distinguish between the two classes.

**Feature Comparisons**
![](images/acoustic-vs-dance.png)
![](images/acoustic-vs-loud.png)

The above graphs show the separability in the data when compared across two unique Spotify features; this suggests that data may separate across an n-dimensional feature space. Given this, the problem can alternatively be posed as an unsupervised learning problem where clustering methods can classify the data.

## Models and Results
Given the unbalanced nature of the dataset, any model chosen would automatically yield high accuracy. So, in addition to aiming for high accuracy, another objective of modeling is to ensure a high AUC (so that TPR is maximized and FPR is minimized). The AUC tells us how well the model is capable of distinguishing between the two classes.

Also, after EDA, I decided to only consider songs released between 2000-2018 because it is evident that music trends and acoustic features change over time, and song characteristics of the '90s would probably be not reflective of '00s and '10s decades. *(Note: For the sake of sample size I decided to combine '00s and '10s decades together. However, with the conglomeration of more songs and awards, it is probably better to consider a smaller time window)*

Here's a list of all the models I tested:
  1. Logistic Regression
  2. Improved Logistic Regression (with un-important Spotify features removed)
  3. LDA
  4. 10-fold CV CART
  5. Random Forest
  6. Bagging
  7. 10-fold CV KNN
  
**Model Summaries:**

| Model   | Accuracy   | TPR   | AUC   |
| -----   | :--------: | :---: | :---: |
| Baseline | 0.798 | na | na |
| Logistic Regression | 0.809 | 0.289 | 0.786 |
| **Improved Logistic Regression** | **0.810** | **0.300** | **0.785** |
| LDA | 0.805 | 0.280 | 0.774 |
| 10-fold CV CART | 0.805 | 0.123 | 0.706 |
| Random Forest | 0.813 | 0.174 | 0.7731 |
| **Bagging** | **0.818** | **0.300** | **0.785** |
| 10-fold CV KNN | 0.801 | 0.014 | 0.736 |

### Additional Modeling

#### Stacking:
Additionally, I tested out an ensemble method by stacking a few models together (logistic + LDA + CART). Model ensembling is a technique in which different models are combined to improve predictive power and improve accuracy. Details regarding stacking and ensemble methods can be found [here](https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html).

**Model Correlation Matrix**

|     | lda | rpart | glm |
| --- | --- | :-----: | :---: |
| **lda** | 1.0000000 | 0.1656148 | 0.9283705 |
| **rpart** | 0.1656148 | 1.0000000 | 0.2025172 |
| **glm** | 0.9283705 | 0.2025172 | 1.0000000 |

**Model Summary**

| Accuracy   | TPR   | AUC   |
| :--------: | :---: | :---: |
| 0.814 | 0.297 | 0.797 |

The stacked model achieved high accuracy and TPR that is comparable to the improved logistic regression and bagging model. However, more importantly, the stacked model greatly improved the AUC.

#### Penalized Regression:
Due to a large number of features (Spotify features + lyrics bag-of-words), I decided to use a penalized logistic regression model. This imposes a penalty to the logistic model for having too many variables. This results in lowering the dimensionality of the feature spacing by shrinking the coefficients of the less important features toward zeros. I specifically used the following penalized regression techniques:

- **Ridge Regression**: all the features are included in the model, but variables with minor contribution have their coefficients close to zero
- **Lasso Regression**: the coefficients of less contributive features are forced to zero and only the most significant features are kept

(an explanation regarding penalty methods and shrinkage can be found [here](https://stats.stackexchange.com/questions/179864/why-does-shrinkage-work))

**Model Summary**

| Model   | Accuracy   | TPR   |
| -----   | :--------: | :---: |
| **Ridge** | 0.805 | 0.182 |
| **Lasso** | 0.807 | 0.185 |


