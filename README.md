# Table of contents

- [Introduction](#introduction)
- [1.0. About the Data](#10-about-the-data)
- [2.0. Hypotheses Creation](#20-hypotheses-creation)
- [3.0. Exploratory Data Analysis](#30-exploratory-data-analysis)
- [4.0. Machine Learning Modeling](#40-machine-learning-modeling)
- [5.0. Business Performance](#50-business-performance)
- [6.0. Next Steps](#60-next-steps)

# Introduction

This project analyzes data from on-line dating application OKCupid.  In recent years, there has been a massive rise in the usage of dating  apps to find love. Many of these apps use sophisticated data science  techniques to recommend possible matches to users and to optimize the  user experience. These apps give us access to a wealth of information  that we've never had before about how different people experience  romance.

The goal of this project is to scope, prep, analyze, and create a machine learning model to solve a question.

**Data sources:**

`profiles.csv` was provided by Codecademy.com.

## Scoping

It's beneficial to create a project scope whenever a new project is  being started. Below are four sections to help guide the project process and progress. The first section is the project goals, a section to  define the high-level objectives and set the intentions for this  project. The next section is the data, luckily in this project, data is  already provided but still needs to be checked if project goals can be  met with the available data. Thirdly, the analysis will have to be  thought through, which include the methods and aligning the question(s)  with the project goals. Lastly, evaluation will help build conclusions  and findings from the analysis.

### Project Goals

In this project, the goal is to utilize the skills learned through  Codecademy and apply machine learning techniques to a data set. The  primary research question that will be answered is whether an OkCupid's user relationship status can be predicted using other variables from their profiles. This project is important since many companies find Valentine's Day an important event for profit.

### Data

The project has one data set provided by Codecademy called `profiles.csv`. In the data, each row represents an OkCupid user and the columns are  the responses to their user profiles which include multi-choice and  short answer questions.

### Analysis

This solution will use descriptive statistics and data visualization  to find key figures in understanding the distribution, count, and  relationship between variables. The project was developed based on CRISP-DS (Cross-Industry Standard Process - Data Science).

### Evaluation

The project will conclude with the evaluation of the machine learning model selected with a validation data set. The output of the  predictions can be checked through a confusion matrix, and metrics such  as accuracy, precision, recall and F1 scores.

# 1.0. About the Data

This dataset analyses user from online dating app OKkCupid. These app give access to a wealth of information that we've never had before about how different people experience romance. 

* The dataset has 59,946 rows and 31 columns.

* The memory usage: 14.2+ MB.

* Dtypes: float64(1), int64(2), object(28).
* No missing values.

### 1.1. Columns Description

##### 1.1.a. VARIABLE DESCRIPTIONS:
Missing data is blank. Some variables have two factors, which are denoted in this codebook by a semicolon (e.g. "graduated from; two-year college) though they are found without punctuation in the dataset. Details for specific variables are found in parenthesis.

##### 1.1.b. Continuous variable
* **age**- User's Age
* **height**- inches
* **income**- (US $, -1 means rather not say) -1, 20000, 30000, 40000, 50000, 60000 70000, 80000, 100000, 150000, 250000, 500000, 1000000.

##### 1.1.c. Categorical variable
- **body_type**- rather not say, thin, overweight, skinny, average, fit, athletic, jacked, a little extra, curvy, full figured, used up
- **diet**- mostly/strictly; anything, vegetarian, vegan, kosher, halal, other
- **drinks**- very often, often, socially, rarely, desperately, not at all
- **drugs**- never, sometimes, often
- **education**- graduated from, working on, dropped out of; high school, two-year college, university, masters program, law school, med school, Ph.D. program, space camp
- **ethnicity**- Asian, middle eastern, black, native American, Indian, pacific islander, Hispanic/Latin, white, other
- **job**- student, art/music/writing, banking/finance, administration, technology, construction, education, entertainment/media, management, hospitality, law, medicine, military, politics/government, sales/marketing, science/engineering, transportation, unemployed, other, rather not say, retire
- **offspring**- has a kid, has kids, doesn't have a kid, doesn't want kids; ,and/,but might want them, wants them, doesn't want any, doesn't want more
- **orientation**- straight, gay, bisexual
- **pets**- has dogs, likes dogs, dislikes dogs; and has cats, likes cats, dislikes cats
- **religion**- agnosticism, atheism, Christianity, Judaism, Catholicism, Islam, Hinduism, Buddhism, Other; and very serious about it, and somewhat serious about it, but not too serious about it, and laughing about it
- **sex**- m, f
- **sign**- aquarius, pices, aries, Taurus, Gemini, cancer, leo, virgo, libra, scorpio, saggitarius, Capricorn; but it doesn’t matter, and it matters a lot, and it’s fun to think about
- **smokes**- yes, sometimes, when drinking, trying to quit, no
- **speaks**- English (fluently, okay, poorly). Afrikaans, Albanian, Arabic, Armenian, Basque, Belarusan, Bengali, Breton, Bulgarian, Catalan, Cebuano, Chechen, Chinese, C++, Croatian, Czech, Danish, Dutch, Esperanto, Estonian, Farsi, Finnish, French, Frisian, Georgian, German, Greek, Gujarati, Ancient Greek, Hawaiian, Hebrew, Hindi, Hungarian, Icelandic, Ilongo, Indonesian, Irish, Italian, Japanese, Khmer, Korean, Latin, Latvian, LISP, Lithuanian, Malay, Maori, Mongolian, Norwegian, Occitan, Other, Persian, Polish, Portuguese, Romanian, Rotuman, Russian, Sanskrit, Sardinian, Serbian, Sign Language, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Tibetan, Turkish, Ukranian, Urdu, Vietnamese, Welsh, Yiddish (fluently, okay, poorly)

##### 1.1.d. Date variable

* **last_online**- last login


##### 1.1.e. Short answer
- **status**- single, seeing someone, married, in an open relationship
- **essay0**- My self summary
- **essay1**- What I’m doing with my life
- **essay2**- I’m really good at
- **essay3**- The first thing people usually notice about me
- **essay4**- Favorite books, movies, show, music, and food
- **essay5**- The six things I could never do without
- **essay6**- I spend a lot of time thinking about
- **essay7**- On a typical Friday night I am
- **essay8**- The most private thing I am willing to admit
- **essay9**- You should message me if...

##### 1.1.f. **SPECIAL NOTES**
All essay questions are fill-in the blank, answers are not summarized here.

### 1.3. Summary statistics:

|        |   count |         mean |          std |  min |  25% |  50% |  75% |       max |     range | unique val. | variation coefficient |    skew | kurtosis |
| -----: | ------: | -----------: | -----------: | ---: | ---: | ---: | ---: | --------: | --------: | ----------: | --------------------: | ------: | -------: |
|    age | 59946.0 |    32.340290 |     9.452779 | 18.0 | 26.0 | 30.0 | 37.0 |     110.0 |      92.0 |          54 |                0.2923 |  1.2658 |   1.5725 |
| height | 59943.0 |    68.295281 |     3.994803 |  1.0 | 66.0 | 68.0 | 71.0 |      95.0 |      94.0 |          60 |                0.0585 | -0.4631 |   7.7611 |
| income | 59946.0 | 20033.222534 | 97346.192104 | -1.0 | -1.0 | -1.0 | -1.0 | 1000000.0 | 1000001.0 |          13 |                4.8592 |  9.0362 |  86.8702 |

* `age` and `income` are high skewed.
* `age`, `height` and `income` has high kurtosis.
* `income` values = -1 are missing values.

For categorical attributes:

| feature | count | unique | frequency |
| ------: | ----: | -----: | --------- |
|  essay0 | 54458 |  54350 | 12        |
|  essay1 | 52374 |  51516 | 61        |
|  essay2 | 50308 |  48635 | 82        |
|  essay3 | 48470 |  43533 | 529       |
|  essay4 | 49409 |  49260 | 16        |
|  essay5 | 49096 |  48963 | 6         |
|  essay6 | 46175 |  43603 | 161       |
|  essay7 | 47495 |  45554 | 89        |
|  essay8 | 40721 |  39324 | 45        |
|  essay9 | 47343 |  45443 | 199       |

* essay attributes we noticed that they have unique inputs, which makes it difficult to analyze.







In this project Machine Learning was used to predict the Status  Relationship and Heights of OkCupid users, setting two different types  of Supervised Machine Learning model, like Regression and  Classification. This is an important feature since many companies profit a lot from Valentine dayes. If users don't input their signs, an  algorithmic solution could have generated a sign to impute missing data  when making matches.

- The regression model for predicting `height` were mostly unsucessful in showing any strong correlation.
- Classification model for predicting `status` showed a  high accuracy, highlighted by K-Neighbors Classifier.



# 2.0. Hypotheses Creation

![hypo](https://github.com/vfamim/Date-a-Scientist/blob/master/img/Relationship_status.png) 

### 2.1. Summary

| ID   | HYPOTHESES                                                   |
| ---- | ------------------------------------------------------------ |
| H1   | Athletic and fit user represent 20% of the total users in relationship status. |
| H2   | Student user represent 50% of the total user in a relationship status. |
| H3   | People who drink less represent 70% of the total user in a relationship status. |
| H4   | Males and females are represented in a balanced way.         |
| H5   | 20% of the total user don't care about sign.                 |
| H6   | Black people represent 30% of the total user in a relationship status. |
| H7   | 10% of user works in health field.                           |
| H8   | 50% of the total user are straight.                          |



# 3.0. Exploratory Data Analysis

### 3.1. Univariate Analysis

Explore the variable so we can check the frequency, distribution, range etc.

#### 3.1.1. Target Variable

![img1](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img01.svg)

#### 3.1.2. Numerical variable distribution

![img2](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img02.svg)

#### 3.1.3. Categorical variable distribution

![img3](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img03.svg)

### 3.2. Bivariate Analysis

The behavior of the variable with respect to the target variable.

##### H1 Athletic and fit user represent 20% of the total users in relationship status.

**TRUE**: Athletic and fit users represents 24.5% of the total user.

![img4](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img04.svg)

##### H2 Student user represent 50% of the total user in a relationship status.

**TRUE**: Students represents 70% of users with a relationship.

![img5](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img05.svg)

##### H3 People who drink less represent 70% of the total user in a relationship status.

**TRUE**: People who drinks less represent 83% of user with a relationship.

![img6](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img06.svg)

##### H4 Males and females are represented in a balanced way.

**TRUE**: Males and females with relationship are balanced.

![img7](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img07.svg)

##### H5 20% of the total user don't care about sign.

**TRUE**: 30.9% of the total user in relationship do not care about signs.

![img8](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img08.svg)

##### H6 Black people represent 30% of the total user in a relationship status.

**FALSE**: Black people represents 1.6% of the total users in a relationship.

![img9](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img09.svg)

##### H7 10% of user works in health field.

**FALSE**: Users who work in health field represents 4.4% of users with relationship.

![img10](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img10.svg)

##### H8 50% of the total user are straight.

**TRUE**: 72.2% f users who are in a relationship consider themselves straight.

![img11](https://github.com/vfamim/Date-a-Scientist/blob/master/img/img11.svg)



# 4.0. Machine Learning Modeling

### 4.1. Models and Performance Metrics

The following models were trained:

* Dummy Classifier;
* K-Nearest Neighbors;
* Random Forest Classifier;
* XGBoost Classifier;

![img_metrics](https://github.com/vfamim/Date-a-Scientist/blob/master/img/1*UVP_xb4F6J-M-xH3haz5Jw.png)

Here is a quick description of the metrics:

- **Accuracy:** is the correct values divided by total values
- **Precision:** is the True Positives divided by the sum of True Positives and False Negatives. So precision is the values of  the true positives divided by the actual positive values.
- **Recall:** is the True Positives divided by the sum of True Positives and False Positives. So recall is the values of the true positives divided by the positive guesses.
- **F1-score:** is a blended score of precision and recall which balances both values.
- **Confusion Matrix:** is a 2x2 matrix that shows the predicted values of  the estimator with respect to the actual values.
  - True Positive (TP): actual positives that are correctly predicted as positive;
  - False Positive (FP): actual negatives that are wrongly predicted as positive;
  - True Negative (TN): actual negatives that are correctly predicted as negative;
  - False Negative (FN): actual positives that are wrongly predicted as negative.

### 4.2. Model Results

#### 4.2.1. Results

The metric chosen will be F1-score.

|           Accuracy | Precision |   Recall |           F1 |      ROC |          |
| -----------------: | --------: | -------: | -----------: | -------: | -------- |
|           Baseline |  0.044663 | 0.000000 | **0.000000** | 0.000000 | 0.500000 |
|        K-Neighbors |  0.791113 | 0.961223 | **0.814193** | 0.881620 | 0.555815 |
|      Random Forest |  0.943885 | 0.955664 | **0.987053** | 0.971105 | 0.503783 |
| XGBoost Classifier |  0.953733 | 0.955683 | **0.997842** | 0.976308 | 0.504049 |

| Baseline: Dummy Classifier                                   | K-Neighbors Classifier                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![ml01](https://github.com/vfamim/Date-a-Scientist/blob/master/img/ml1.svg) | ![ml02](https://github.com/vfamim/Date-a-Scientist/blob/master/img/ml2.svg) |

| Random Forest Classifier                                     | XGBoost Classifier                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![ml3](https://github.com/vfamim/Date-a-Scientist/blob/master/img/ml3.svg) | ![ml4](https://github.com/vfamim/Date-a-Scientist/blob/master/img/ml4.svg) |

#### 4.2.2. Real results: Cross-Validation

All cross-validation use 10-fold.

|            test_Accuracy |  test_Precision |     test_Recall |         test_F1 |        test_ROC |                 |
| -----------------------: | --------------: | --------------: | --------------: | --------------: | --------------- |
|              K-Neighbors | 0.9008+/-0.0089 | 0.9845+/-0.0081 | 0.8144+/-0.0162 | 0.8914+/-0.0104 | 0.9008+/-0.0089 |
| Random Forest Classifier | 0.9774+/-0.0869 | 0.9749+/-0.1389 | 0.9861+/-0.0067 | 0.9792+/-0.0756 | 0.9774+/-0.0869 |
|       XGBoost Classifier | 0.9763+/-0.1356 | 0.9692+/-0.1907 | 0.9965+/-0.0045 | 0.9802+/-0.1112 | 0.9763+/-0.1356 |

### 4.3. Fine Tuning: Grid Search

Grid search is the process of performing hyper parameter tuning in order to determine the optimal values for a given model. This is significant as the performance of the entire model is based on the hyper parameter values specified. There are libraries that have been implemented, such as `GridSearchCV` of the `sklearn` library.

#### 4.3.1. XGBoost Classifier Tuned Performance

|                      | Accuracy | Precision |   Recall |       F1 | ROC     |
| -------------------: | -------: | --------: | -------: | -------: | ------- |
| XGBoost Classifier + | 0.953733 |  0.955683 | 0.997842 | 0.976308 | 0.50404 |

![ml_final](https://github.com/vfamim/Date-a-Scientist/blob/master/img/ml_final.svg)

### 4.3.2. Cross Validation Final Model

|                      |   test_Accuracy |  test_Precision |     test_Recall |         test_F1 |        test_ROC |
| -------------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| XGBoost Classifier + | 0.9763+/-0.1356 | 0.9692+/-0.1907 | 0.9965+/-0.0045 | 0.9802+/-0.1112 | 0.9763+/-0.1356 |



# 5.0. Business Performance

### 5.1. Final Results

|             Accuracy | Precision |   Recall |       F1 |      ROC |          |
| -------------------: | --------: | -------: | -------: | -------: | -------- |
|             Baseline |  0.044663 | 0.000000 | 0.000000 | 0.000000 | 0.500000 |
|          K-Neighbors |  0.791113 | 0.961223 | 0.814193 | 0.881620 | 0.555815 |
|        Random Forest |  0.943885 | 0.955664 | 0.987053 | 0.971105 | 0.503783 |
|   XGBoost Classifier |  0.953733 | 0.955683 | 0.997842 | 0.976308 | 0.504049 |
| XGBoost Classifier + |  0.953733 | 0.955683 | 0.997842 | 0.976308 | 0.504049 |

### 5.2. Final Real Results

|                          |   test_Accuracy |  test_Precision |     test_Recall |             test_F1 |        test_ROC |
| -----------------------: | --------------: | --------------: | --------------: | ------------------: | --------------: |
|      Logistic Regression | 0.9008+/-0.0089 | 0.9845+/-0.0081 | 0.8144+/-0.0162 | **0.8914+/-0.0104** | 0.9008+/-0.0089 |
| Random Forest Classifier | 0.9774+/-0.0869 | 0.9749+/-0.1389 | 0.9861+/-0.0067 | **0.9792+/-0.0756** | 0.9774+/-0.0869 |
|       XGBoost Classifier | 0.9763+/-0.1356 | 0.9692+/-0.1907 | 0.9965+/-0.0045 | **0.9802+/-0.1112** | 0.9763+/-0.1356 |
|     XGBoost Classifier + | 0.9763+/-0.1356 | 0.9692+/-0.1907 | 0.9965+/-0.0045 | **0.9802+/-0.1112** | 0.9763+/-0.1356 |

# 6.0. Next Steps

1. Model deployment;
2. New features;
3. Model's hyper parameters experiment and evaluation.

