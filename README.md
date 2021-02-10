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

![hypo]() 

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