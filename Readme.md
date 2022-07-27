# Objective and selection of User engagement metric

Our objective is to predict user engagement for a particular video for a particular viewer, I have decided to take 'watched_pct' as the viewer engagement metric.

- **Advantages:**
	- Watched percentage is not dependant on the length or type or other features of a video. Hence this feature can generalize the user engagement acorss several publishers and users.
- **Disdvantages:**
	- Watched percentage does not take into account the user interaction with the video, like if an ad were skipped. Hence this metric does not capture whether the user is actively watching the video.
	- Another metric potentially could be 'watched duration', however the disadvantage would be different videos have different length and the shorter videos could have potentially higher user engagement compared to longer videos.

# Model Implementation

- The input csv file should be placed in the same folder as the ipynb file containing the model.
- Exploratory data analysis:
	- **Check for missing values:** First check is to determine if there are columns which has null values. If so, then a method should be designed on how to handle those missing values depending on whether they numerical or categorical columns. Based on the analysis performed, there is only column country_code which has 11 missing values. Since, this is a categorical columns the missing values are imputed by the mode of the column.
	- **Drop columns which are not useful:** Several columns with id's like viewed_id, embed_id and media_id are not useful in the prediction modelling as they are just id's. They are dropped from the dataset.
	- **Processing Numerical Variables:**
		- There are in total 6 numerical columns. The numerical columns contain both discrete and continous variables.
		- In this case there is only one discrete variable which is 'viewer_tz_offset'.
		- It is checked if there is any noticeable relationship between this discrete variable and the prediction variable (watched_pct). After creating a plot, there are no big noticeable relationship between them.
		- There are 5 continous variables in the Numerical variables.
		- Plot the continous variable against the watched_pct to determine if there is any noticeable relationship with the prediction variable. In this situation, the data is quite skewed to the right with most of the columns having lot of zeroes.
		- An effort is made to transform these skewed data with log normal and sqrt transformation to see if there can be a linear relationship between the features and the prediction variable.
		- No linear relationship can be found from the pairplot created for these numerical continous variables.
		- After reading some existing research papers and investigation, It is adviseable to not do a linear regression when most of the numerical features have zeroes and heavily skewed.
		- However a linear regression can be fitted (for trial) as even if the individual features does not linearly correlate with the prediction variable, maybe they together will. Once a linear regression model is created,  the assumptions for a linear regression model like the residuals follow a normal distribution can be validated.
		- However, here I have decided that to proceed with tree based models as they work quite well with skewed data. Hence did not pursue the linear regression option.
	- **Processing start time variable to convert them to seconds from midnight:**
		- Since the start time variable contains temporal information, it is better to convert them to an offset in seconds from midnight.
	- **Analysis of Categorical variables:**
		- Identify all the categorical variables and in this case there are 6 of them.
		- A check is made to determine how many unique categories are available for each of the categorical columns
		- Identify if there are any visible relationship between these variables and the watched_pct. Based on the plots, there are no noticeable big trends or relationships visible.
		- Since the boosting models (except I think catboost) accepts only numerical values, I have one hot encoded them. This is required to be done before the train test split to ensure all the one hot encoded categories are present in both train and test.
		- Finally the Categorical variables are one hot encoded using pd.get_dummies().
	- **split the data into train and test dataset:**
		- In order to validate the performance of the trained model on unseen data, the dataset is divided into train and test data with 10% of data being part of the test dataset
	- **Model Training:**
		- I have chosen three Tree based regression models namely
			- XGBoost regressor based on boosting method
			- Decision Tree regressor
			- Random Forest regressor based on bagging method.
			**Advantages of Tree based models:**
				- Works very well with large datasets
				- features need not be normalized as our features are on different scales 
				- works well when the data is non linear which is in our case
				- works very well with skewed data and our data from the exploration phase we have determined that they are very skewed. 
				- Explainable model based on feature importance
			**Disadvantages of Tree based models:** 
				- These models tend to overfit on the data especially when you create a very deep tree and with noisy data especially if your hyper parameters are not tuned properly. 
				- They could be very sensitive to outliers especially with boosting as the current learner learns based on the errors made by the previous learner. 
				- The training time is higher than for eg: a linear regression or a decision tree for the XGboost and Randomforest models (ensembling models).
		- I have used Randomized search CV for each of these models to determine the best parameters to be used for fitting the training data.
	- **Model evaluation and final model:**
		- We can use many evaluation metric for the regression model like MAE, MSE, RMSE etc..I have used MAE as the error metric since the data has outliers and I want to include them in the loss function. Addtionally, since MAE will represent the impact propotional to the actual increase in error which could be relevant here especially since we are measuring user engagement, whereas in RMSE this would be squared and hence higher which could be very relevant in critical applications like critical trials etc.
		- The MAE is quite small for both the training and test set for all the three models. Since Randomforest regressor has an MAE of zero on the test dataset, I will choose this model as the final model to be saved via pickle for future predictions.
	- **Further improvements to the model:**
		- The randomforest regressor is not overfitting on the current data based on the low MAE also on the test data. The model needs to be validated further with more data to ensure it is not overfitting. If it is overfitting, then model parameters needs to be tuned further.
		- Implement feature selection based on their importance to determine if some of them can be excluded from model training.
		- Lastly, we can add more features which could potentially influence the user engagement like likes, shares, comments (not sure though if they are accessible data as they are outside the player), whether the user has any of the features in the player like stop, FF or backward buttons etc, whether the user has watched on full screen or minimized, etc..

# Hosting the model to provide real time predictions:
	- We can create API's using Flask libraries to run the prediction on the saved model. An example code on how this can be done is below:
		from flask import Flask, request
		import pickle
		
		app=Flask(__name__)
		read_model=open('User_engagement.pkl','rb)
		regressor=pickle.load(read_model)
		
		@app.route('/predict')
		def prediction():
			play_seq=request.arg.get('play seq')
			....
			...
			..
			prediction=regressor.predict([[play_seq,...]])
			return prediction
		
		if __name__=='__main__':
			app.run()
	- We can containerize this application using docker and register them as container in any of the cloud application in the respective container registery.
	- Once the container is registered, then the application can be deployed in Kubernetes in the corresponding cloud for scalability.
	- The application with the flask can be trigerred using the url (API) exposed by the kubernetes cluster to generate the predictions which can be provided real time to the users.