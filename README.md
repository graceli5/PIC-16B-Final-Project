# PIC16B_FinalProject
Repo for PIC 16B group project @ UCLA during the spring 2024 quarter.

# Problem Proposal

This project will explore the current United States real-estate market, investigate what factors influence the price of property, and create multiple machine learning models that predict these housing costs throughout the country. This topic is very relevant to our present day lives, as we are students that pay to live in housing off-campus and will eventually be looking to move to various cities post-college. Being able to infer and understand the trends of real estate is extremely valuable economic knowledge that will provide important insights about our country.

We are also curious about what underlying societal factors such as average income, political alignment, or quality of schools for example affect the prices of houses. These concepts tie in to general education humanities courses we have taken at UCLA that explore the relationship between societal factors and external structures like the economy. Additionally, we aim to focus a major part of the complexity of our project on the machine learning aspect of our data. We are data science majors and minors, and are excited to gain valuable practice with testing different machine learning models and evaluating their accuracy because this closely ties into our academic and career aspirations.

# Required Resources

#### Data: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

Our main dataset (linked above) contains over 2.2 million entries of housing listings on the market in the United States, that have been collected starting in April 2022 and updated weekly through present day. These listing entries contain the locations of the houses, house area/size, information regarding dates it was previously sold, number of bedrooms, and other attributes that are physically related to the houses. To expand upon this data and investigate underlying societal factors of real estate, we will join this main dataset with others.

Our dataset starts with 12 columns. In order to gain a better understanding of what we are working with, an outline of the meaning and statistical details of each of these features is below.

- brokered_by: float, agency that is selling/sold property label encoded numerically for privacy
- status: string, either "for_sale" or "sold" representing the status of the property on the market
- price: float, price of property listing/sale rounded to the nearest dollar
- bed: float, number of bedrooms
- bath: float, number of bathrooms (to the nearest whole number, "half baths" are counted as full)
- acre_lot: float, total lot (quantity of land) size in square acres rounded to 2 decimal places
- street: float, street of property address label encoded numerically for privacy
- city: string, name of the city the property resides in
- state: string, name of the state or territory the property resides in, capitalized
- zip_code: float, zip code the property resides in
- house_size: float, house/living space size in square feet, rounded to the nearest integer
- prev_sold_date: year-month-day, contains date it was sold if house was previously registered on the market


#### Data Cleaning and Merging:

Processing our data to make it cleaner for use included taking a random subset of the dataframe, dropping unnecessary features, and removing missing entries. Firstly, since our dataset has millions of entries and we would like to run machine models on it, we decided to trim it down to fewer samples in order to save computational cost. After experimentation, we decided on taking a random sample of 50,000 entries, as this struck a nice balance between allowing our models to run in a reasonable timeframe and still containing plenty of entries to allow significant conclusions (even after future data trimming).

Additionally, we dropped a few features that we found unnecessary in informing our predictions, specifically the columns of brokered_by, status, prev_sold_date, and street. A justification for removing each of these columns follows: 

- We are not interested in the real-estate company in charge of each listing and prefer to look at actual qualities of the house/area, so we dropped brokered_by.
- Status is removed because we will treat the price of the house the same regardless of whether it is for sale or sold.
- Prev_sold_date can be dropped because we are only focused on the current selling price and characteristics of the house.
- Street is an extremely niche measure of location, and the zip code/state features provide us with more broad and informative groupings of samples so we prefer these.

Along with these feature columns, we also removed some sample rows that did not align with our project goals. This included any entries that came from Puerto Rico or the Virgin Islands because our project will focus on just USA states, not territories as well. We also had to decide how to deal with some missing values. Upon analyzing how many were contained in our dataset, we concluded there were few enough that we could just remove any rows containing NaN values and we would still maintain an adequate sized dataset. After doing this along with all of the above steps, we were left with a clean dataset with only the features we needed, every entry being complete with data, and a sample of over 30,000 listings to work with.

Once our data was trimmed down through cleaning, we built it back up by merging with other datasets in order to add new feature columns. Recall that one of our goals was to investigate what factors may influence housing prices, we wanted to expand upon previous work done on this and elevate it by including new features to analyze. One important thing to note is that we ensured while we were gathering new data to merge that it was sampled from a similar time period to our current dataset (the past few years). We merged with three other datasets, the first being "US Household Income by Zip Code 2021-2011" found at Kaggle. This dataset contains national census results from the past two decades, and we took only samples from 2021 as this aligns with the timeframe of price data we are pulling from. We merged the feature of “Nonfamily median household income” - which contains a float rounded to the nearest dollar of each zip code’s median income of non-marriage reports- with our dataset along the common column of zip code using an inner merge. There were barely any NaN entries present, so we just removed the few rows that contained them.

We were also interested in how some more “outside the box” societal factors will correlate to housing prices, so we merged with Kaggle’s "2020 US Presidential Election Results by State" to investigate political affiliation. The columns we used were “biden_pct” and “trump_pct”, which contained a float of the percentage of popular vote in every state cast towards the respective candidate during the 2020 presidential election, rounded to one decimal place. This was merged along the column of state, and the dataset was already very clean so it did not require additional processing. Lastly, we gathered minimum wage data from Kaggle’s "Living Wage - State Capitals" dataset which contained a column of the current minimum wage of every state capital as a float expressed to the cent. This again did not contain any missing values, so merging with our dataset along the column of state proved simple. 

The three datasets we merged with are provided below for more detail:

#### Income by zipcode- 
- https://www.kaggle.com/datasets/claygendron/us-household-income-by-zip-code-2021-2011

#### Presidential election results-
- https://www.kaggle.com/datasets/callummacpherson14/2020-us-presidential-election-results-by-state

#### Minimum wage-
- https://www.kaggle.com/datasets/brandonconrady/living-wage-state-capitals

# Prior Works Associated 

https://www.kaggle.com/code/binfeng2021/regression-problem-house-price-prediction

Most of the previous works that have to do with this data set are in regards to data visualization. There are a few public notebooks that investigate the correlation between column variables such as number of acres, bedrooms, etc with home prices, and graph the results using various visualization techniques. We will aim to dive beyond this current work by merging our dataset with new data, in order to add more abstract variables to analyze the home prices trends with. This will create new material to visualize in creative ways, and draw more insightful conclusions from. 

There are limited prior works on this data set regarding machine learning. A couple employ linear regression and look at a mean squared error, but they conclude that the data does not fit a linear pattern very conclusively. There are also better results with a random forest regressor and gradient boosting regressor which provide R^2 values of 0.9 and 0.76. 
Our goal is to expand on this previous work by employing more complex machine learning techniques, and possibly deep learning techniques in order to refine our predictive model and obtain a high accuracy value.

# Required Tools and Skills

Knowledge of the sklearn library will be required for this project to conduct regressions and ML algorithms to make predictions regarding real estate costs based on the original variables in the real estate data (which are more literal about the house e.g., location, acres, number of rooms and bathrooms). 

Beyond these basic attributes of the home, we wish to expand upon the real estate data and join/merge it by zip code or state, using manipulation tricks in the pandas library. Some of the features we wish to add include political party, school quality, age of population, quality of life, income demographics, population density of zip code, etc.This will allow our project to expand beyond the physical features of the home, and attempt to make connections to housing prices in relation to socioeconomic factors in the areas. 
Once we widen the data using joins, we will test if adding these features improves the accuracy of the model. Additionally, PCA and/or other dimension reduction methods will be utilized to remove uninformative features from the model. 

In addition to model building, since the housing data is geographical, we will expand upon our current visualization skills (using plotly and plotly.express) to build interesting maps to visualize locations of real estate, and color by features that produce compelling insights. We plan to attempt to employ geoplot, which is a high-level geospatial plotting library, which will allow us to create even more interesting visualizations. 

For more basic graphics and visualizations (comparing prevalence of categories or densities of the numerical predictors, etc.) we will leverage libraries such as seaborn and matplotlib. 



# Project Summary

Our project was driven by three main goals. The first of which was to identify the factors that  influence– and most heavily influence– the housing market. Beyone some more obviously correlated features (for example bedrooms, lot size, etc) that deal with physical properties of the house, we also investigated the impact of societal factors of the surrounding area such as average income and minimum wage. Our second goal was to evaluate and compare multiple machine learning models to identify which most effectively classify a house as expensive or inexpensive based on its features. This included developing SVM, Logistic regression, K nearest neighbors, decision tree, random forest, max voting, and neural network models and testing their effectiveness. Our third goal, building off the previous one, was to develop a model that obtains a high accuracy score in classifying houses as expensive or inexpensive. Previous works found using our same datasets focusede on either just visualizations or a different target column so we do not have a strong baseline to compare to, but we prioritized achieving an accuracy score in the high 80 percents, as this is realistically achievable yet still indicates our model is a strong predictor.

Utilizing GridSearchCV, we were able to find the opitmal parameters for each ML model and determine the 'best' hyperparameters and number of hidden layers and neurons for the NN using all 8 of our numerical features. The most optimal NN model had the following attributes:

#### Layers
- Input neurons: 8 (8 features)
- Hidden layer #1: 6 neurons
- Hidden layer #2: 32 neurons
- Dropout layer: p = 0.2
    - The dropout value of 0.2 was used to help prevent overfitting.
- Output layer: 1 (binary 0 or 1 for above or below median price for that state)

#### Hyperparameters for the Hidden Layers
- activation = 'relu' selected as the best activation function (rather than 'sigmoid').
- kernel_initializer = "he_normal" selected since it tends to pair effectively with reLU activation.
- loss: binary cross entropy.
- optimizer = 'adam' was selected for its general success on other models and wide utilization.
- metrics: accuracy.
- epochs = 50 was selected since our machines had the ability and efficiency to run a larger number, which will hopefully help decrease the training loss.
- batch_size = 16 was selected as the optimal batch size (coompared to 32).

### NN Model Performance
The most optimal NN loss leveled out by 50 epochs at 0.43 for both training and validation, indicating the NN was "fully trained". The achieved training accuracy was 78.57% and the validation accuracy was 77.37%. Since these are similar, it indicated that the model is not prone to overfitting. This was further confirmed as applying this NN to the testing data achieved a testing accuracy of 78.93% which is similar to the training accuracy, and even slightly better than the validation accuracy. This suggests the model is generalizable on unseen data. 

Results from 20-Fold cross validation of the model found:
- The minimum accuracy across 20 folds of the testing data is 74.9 %
- The maximum accuracy across 20 folds of the testing data is 83.13 %
- The average accuracy across 20 folds of the testing data is 78.3 %
- The the standard deviation of accuracy across 20 folds of the testing data is 0.02

Although results from the 20-fold CV were positive and indicated the mode is consistent and can perform well on unseen data, in terms of accuracy, the ML random forest classifier outperformed the NN. 

### Random Forest Model Performance

Although we explored models with varying batch size, hidden layers, and activation functions using grid search as well as our custom threshold in order to optimize our neural network, it still outputs a lower testing accuracy 79.56%  than some of our other "simpler" machine learning models. In fact, our final choice of ‘best’ model is the ML random forest model because it achieved the highest testing accuracy score of 81.22%. 

Our random forest model was fitted over 5 folds for each of the 10 candidates, totalling 50 fits. The best parameters selected were:

- n_estimators = 200 trees in the forest.
- max_features = ‘sqrt’.
- max_depth = None (no maximum depth of the tree).
- criterion = ‘entropy,’ the function to measure the quality of a split for information gain.

Employing this model on our housing data to classify whether a house price falls above or below the median price per that state given our 8 features, a training accuracy of 81.7% and a similarly high testing accuracy of 81.22% were achieved. Since one of our main goals was to ultimately select the model that achieves the highest accuracies, the random forest model was selected. 


# Project Findings and Conclusion 

We are overall very happy with the success of our project and satisfied in meeting the majority of our goals. Our work has many advantages including creating a model that can be applied to real-world and current day data. Since our dataset is extremely recent and all scraped from the past two years, it captures a very present-day snapshot of the United States housing market. Many other similar datasets on Kaggle and their respective models draw on more dated samples from a few decades, which makes predictions less-applicable to solving current-day problems. Additionally, when compared to previous work on the same Kaggle dataset that we used, our model did a nice job expanding on the features by merging with other data. This meant that our visualizations and models were able to factor in variables like median zip code income, and minimum wage which others did not, and they did prove informative. Additionally, most prior work on this dataset focused on visualizations more than predictions, and those that employed models did so with just one or two rather than creating many to compare like we did. Our analysis of over six models complete with cross-validation provided us with very comprehensive and thorough conclusions. This was important to back up our accuracy scores achieved by the random forest ML model, to ensure that it wasn’t a “fluke” that our best model was performing as such.

However, there are a few limitations and areas we might look to expand upon in the future. For one, our accuracy could be improved in order to apply to real-world scenarios. While low 80s accuracy clearly shows that our model has learned from our data and is much more effective then randomly guessing, we would have liked to be able to improve to high 80s or low 90s. If this model were to be built to be used by a business for instance, we would want a much lower error rate in order to ensure appropriate results. Additionally, our model is limited in the fact that it classifies price rather than predict it numerically. Guessing a specific price number might be more useful in some cases, and this is a route we could possibly explore in the future instead.
