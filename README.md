# Machine-Learning-With-Python

## What is it? , How it works? and When to use the Model

### What is Naive Bayes Model

Naive Bayes Model is an old method for classification and predictor selection that is enjoying a renaissance because of its simplicity and stability.

Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.It is widely use for text classification which used in various applications google search, email sorting, language detection.

Below is the formula for calculating the conditional probability.

P(H|E) = P(E|H) * P(H)/P(E)

where

P(H) is the probability of hypothesis H being true. This is known as the prior probability.  
P(E) is the probability of the evidence(regardless of the hypothesis).  
P(E|H) is the probability of the evidence given that hypothesis is true.  
P(H|E) is the probability of the hypothesis given that the evidence is there.  

### How it works

Fruit Example

Let's try it out on an example to increase our understanding: The OP asked for a 'fruit' identification example.

Let's say that we have data on 1000 pieces of fruit. They happen to be Banana, Orange or some Other Fruit. We know 3 characteristics about each fruit:

Whether it is Long
Whether it is Sweet and
If its color is Yellow.  
This is our 'training set.' We will use this to predict the type of any new fruit we encounter.  

Type           Long | Not Long || Sweet | Not Sweet || Yellow |Not Yellow|Total  
             ___________________________________________________________________  
Banana      |  400  |    100   || 350   |    150    ||  450   |  50      |  500  
Orange      |    0  |    300   || 150   |    150    ||  300   |   0      |  300  
Other Fruit |  100  |    100   || 150   |     50    ||   50   | 150      |  200  
            ____________________________________________________________________  
Total       |  500  |    500   || 650   |    350    ||  800   | 200      | 1000  
             ___________________________________________________________________  
             
We can pre-compute a lot of things about our fruit collection.  

The so-called "Prior" probabilities. (If we didn't know any of the fruit attributes, this would be our guess.) These are our base rates.

 P(Banana)      = 0.5 (500/1000)  
 P(Orange)      = 0.3  
 P(Other Fruit) = 0.2  
Probability of "Evidence"  

p(Long)   = 0.5  
P(Sweet)  = 0.65  
P(Yellow) = 0.8  
Probability of "Likelihood"  

P(Long|Banana) = 0.8  
P(Long|Orange) = 0  [Oranges are never long in all the fruit we have seen.]  
 ....

P(Yellow|Other Fruit)     =  50/200 = 0.25  
P(Not Yellow|Other Fruit) = 0.75  
Given a Fruit, how to classify it?  

Let's say that we are given the properties of an unknown fruit, and asked to classify it. We are told that the fruit is Long, Sweet and Yellow. Is it a Banana? Is it an Orange? Or Is it some Other Fruit?

We can simply run the numbers for each of the 3 outcomes, one by one. Then we choose the highest probability and 'classify' our unknown fruit as belonging to the class that had the highest probability based on our prior evidence (our 1000 fruit training set):

P(Banana|Long, Sweet and Yellow)   
       P(Long|Banana) * P(Sweet|Banana) * P(Yellow|Banana) * P(banana)  
    = _______________________________________________________________  
                      P(Long) * P(Sweet) * P(Yellow)  

    = 0.8 * 0.7 * 0.9 * 0.5 / P(evidence)  
 
    = 0.252 / P(evidence)  


P(Orange|Long, Sweet and Yellow) = 0  


P(Other Fruit|Long, Sweet and Yellow)  
      P(Long|Other fruit) * P(Sweet|Other fruit) * P(Yellow|Other fruit) * P(Other Fruit)  
    = ____________________________________________________________________________________  
                                          P(evidence)

    = (100/200 * 150/200 * 50/200 * 200/1000) / P(evidence)

    = 0.01875 / P(evidence)
By an overwhelming margin (0.252 >> 0.01875), we classify this Sweet/Long/Yellow fruit as likely to be a Banana.  

Why is Bayes Classifier so popular?  

Look at what it eventually comes down to. Just some counting and multiplication. We can pre-compute all these terms, and so classifying becomes easy, quick and efficient.  

Let z = 1 / P(evidence). Now we quickly compute the following three quantities.  

P(Banana|evidence) = z * Prob(Banana) * Prob(Evidence1|Banana) * Prob(Evidence2|Banana) ...  
P(Orange|Evidence) = z * Prob(Orange) * Prob(Evidence1|Orange) * Prob(Evidence2|Orange) ...  
P(Other|Evidence)  = z * Prob(Other)  * Prob(Evidence1|Other)  * Prob(Evidence2|Other)  ...  
Assign the class label of whichever is the highest number, and you are done.  

### When to use this model

- Real time Prediction: Naive Bayes is an eager learning classifier and it is sure fast. Thus, it could be used for making predictions in real time.
- Multi class Prediction: This algorithm is also well known for multi class prediction feature. Here we can predict the probability of multiple classes of target variable.
- Text classification/ Spam Filtering/ Sentiment Analysis: Naive Bayes classifiers mostly used in text classification (due to better result in multi class problems and independence rule) have higher success rate as compared to other algorithms. As a result, it is widely used in Spam filtering (identify spam e-mail) and Sentiment Analysis (in social media analysis, to identify positive and negative customer sentiments)
- Recommendation System: Naive Bayes Classifier and Collaborative Filtering together builds a Recommendation System that uses machine learning and data mining techniques to filter unseen information and predict whether a user would like a given resource or not

