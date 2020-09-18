ChillPill(V1): Find antidepressants as unique as you are
==============================

 Antidepressants can be a complete life-saver to as many as 19 million Americans each year. However, they can come with 
 side effects that patients may not be aware of when they begin their treatment. This is especially hard to navigate as 
 a first time patient, contributing to about 30% of them having to swtich between 2~3 different drugs until they find the one
 that works for them. Existing platforms, such as Google or webMD are not good at capturing the prevalence of side 
 effects. For example, if I ask google about the side of Prozac, it'll tell me I may get insomnic, but how likely it is,
  is the question that I have as a patient. To tackle this problem, I created ChillPill, a web app that recommends 
  anti-depressants based on patients' symptoms and the most common side effects from taking pills so that patients can 
  further consult with their doctors.
   

Example use case is the following: Suppose that I've struggled with hyperinsomnia and social anxiety throughout my life 
and I'm in search for a pill that works. Once I type in my symptoms and click submit, the app returns two results; first
 is the ranked ordered list of drugs that are likely to give me positive outcomes, based on other patients' experience, 
 who've had the similiar symptoms as mine. In addition, the app shows the 10 most common side effects experienced by 
 patients, as reported in their language as opposed to medical jargon.
     

 The project was made possible by data sources that are public and free. To build the recommendation, I used sub reddit 
 channel on anti-depressants where patients exchange information and their experiences. Subreddit channel contains rich 
 information about patients' experience at various points of treatment cycle. For the side effect, I used structured 
 survey data from AskPatients.com in addition to subreddit comments.

 ***Methods***

 The app implements two major components: recommendation and the prevalence of side effect.

 The algorithm for recommendation is in essence a combination of text classification and sentiment analysis. The goal here 
 is to predict pills that will give positive outcomes, given new user's symptoms. Therefore, we want our classfier to 
 learn what words are often used in association with a drug, to link symptoms to drugs. In addition, classifier also has
 to know the context in which those words are used - specifically, whether the drugs alleviated the symptoms or not. In 
 the area of Natural Language processing, understanding the context and intentions of the sentence remains as an active
 area of research, and there is no silver bullet algorithm that fits all use cases. Here, I used the sentiment analyzer
 from nltk package, which was pretrained on large corpus of news paper article, to classify sentiment of given document 
 into three categories: "positive", "neutral", "negative". Then, I gave scores for each comment, 5 for positive, 3 for 
 neutral and 1 for negative. These scores were used to bias the learning of classifier so that they prioritize fitting 
 words associated with positive outcomes. Resulting classifier can now predict which drug is likely to work for patients
  given their symptoms


 Side effect prevalence was measured in two step: firstly, I extracted keywords of side effects from Ask Patient survey 
 data. The intention was to capture the patients' vocabulary in describing their side effects, as opposed to medical 
 jargon used in WebMD or google search. These keywords were then searched through the larger corpus of the subreddit 
 comment to gauge their prevalence.

 This app was created during my fellowship at Insight Datascience program. Creating the app within 3 weeks was no less 
 than a daunting task however, was absolutely a gratifying experience. Having survived depression, I tried to answer 
 what I and my peers may find most useful in creating and designing the app. Having said that, this project has a lot of
  room to grow: The biggest challenge came from separating words describing symptoms from other thematic words. This 
  matters because these words are features for training the recommendation and affect the accuracy. Another challenge 
  came from the way that I extracted the side effects. I've used an unsupervised learning method. However, reflecting 
  that it requires pre and post processing, I think it'll be worthwhile to train binary classifer that tells whether 
  a sentence is telling a side effect or not.

I've shut down the hosting for the webapp after the program but find out more in this presentation.
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRnCquSOvsO4ohdTlVoQzvcCmMqOEHrJLWuEtYgGF6sBHEPtmWecvbBwpbDF9Gcek4m0xNa0f6yf82S/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

Project Organization
------------
Under Jupyernotebooks, you'll find notebooks for
1. Data collection from Reddit corpus
2. EDA on gathered data from Reddit and building classifier for most-talked about anti-depressants
3. Side effect information processing
    - Topic extraction from survey corpus (Askpatient.com)
    - Quantifying frequency of gathered side effect topics from above 
    - Visualizing the frequency of the side effects 
Under src, you'll find scripts of utility functions used across notebooks 

Project Tree
------------

    ├── LICENSE
    ├── Makefile           
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
