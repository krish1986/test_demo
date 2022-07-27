# Objective and selection of Use engagement metric

Our objective is to predict user engagement for a particular viewer for a particular viewer, I have decided to take 'watched_pct' as the viewer engagement metric.

**Advantages:

Watched percentage is not dependant on the length or type or other features of a video.

**Disdvantages:

Watched percentage does not take into account the user interaction with the video, like if an ad were skipped. Hence this metric does not capture whether the user is actively watching the video.
Another metric potentially could be 'watched duration', however the disadvantage would be different videos have different length and the shorter videos could have potentially higher user engagement compared to longer videos.

# Model Implementation

- The input csv file should be placed in the same folder as the ipynb file containing the model.
- Exploratory data analysis:
	- Check for missing values: First check is to determine if there are columns which has null values. If so, then a method should be designed on how to handle those missing values depending on whether they numerical or categorical columns.After executing the below steps of code, only the column country_code, there are 11 missing values. Since, they are categorical columns, I have decided to impute the missing values by the mode of the column.
	- Drop columns which are not useful: Several columns with id's like viewed_id, embed_id and media_id are not useful in the prediction modelling as they are just id's. They can be dropped.
	- Processing Numerical Variables:
		- Determine which columns are numerical columns and observe them. They can be both discrete and continous variables. Here there are in total 6 numerical columns.
		- Determine the discrete variable, which in this case is only 'viewer_tz_offset'
		- Validate if there is any noticeable relationship between this discrete variable and the prediction variable. In this situation, there are not big noticeable relationship between them.
		- Determine the continous variables
		- Plot the continous variable against the watched_pct to determine if there is any noticeable relationship with the prediction variable. In this situation, the data is quite skewed to the right with most of the columns having lot of zeroes.
		- An effort is made to transform these skewed data with log normal and sqrt transformation to see if there can be a linear relationship between the features and the prediction variable.
		- No linear relationship can be found from the pairplot.
		- After reading some existing research papers, It is adviseable to not do a linear regression when most of the numerical features have zeroes and heavily skewed.
		- However a linear regression can be fitted (for trial) to determine if the assumptions for a linear regression model like the residuals follow a normal distribution can be validated.
		- However, here I have decided that Tree based models work quite well with skewed data and I have decided to proceed forward with them.
	- Processing start time variable to convert them to seconds from midnight:
		- Since the start time variable contains temporal information, it is better to convert them to an offset in seconds from midnight.
	- Analysis of Categorical variables:
		- Identify all the categorical variables and in this casen there are 6 of them.
		- A check is made to determine how many unique categories are available for each of the categorical columns
		- Identify if there are any visible relationship between these variables and the watched_pct. Based on the plots, there are no noticeable big trends or relationships visible.
		- Since the boosting models (except I think catboost) accepts only numerical values, I have on hot encoded them. This is required to be done before the train test split to ensure all the one hot encoded categories are present in both train and test.
		- Finally the Categorical variables are one hot encoded using pd.get_dummies().
	- split the data into train and test dataset:
		- In order to validate the performance of the trained model on unseen data, the dataset is divided into train and test data with 10% of data being part of the test dataset
	- Model Training
		- I have chosen three Tree based regression models namely
			- XGBoost regressor based on boosting method
			- Decision Tree regressor
			- Random Forest regressor based on bagging method.
		- I have used Randomized search CV for each of these models to determine the best parameters to be used for fitting the training data.
			** Advantages of Tree based models: The main reson why I choose XGBoost is it works very well with large datasets, features need not be normalized as our features are on different scales and also works well when the data is non linear which is in our case. These algorithms also work very well with skewed data and our data from the exploration phase we have determined that they are very skewed. It is also an explainable model based on feature importance.
			** Disadvantages of Tree based models: These models tend to overfit on the data especially when you create a very deep tree and with noisy data especially if your hyper parameters are not tuned properly. They could be very sensitive to outliers especially with boosting as the current learner learns based on the errors made by the previous learner. The training time is higher than for eg: a linear regression or a decision tree for the XGboost and Randomforest models.
		- We can use many evaluation metric for the regression model like MAE, MSE, RMSE etc..I used MAE as the error metric since the data has outliers and I want to include them in the loss function. Addtionally, since MAE will represent the impact propotional to the actual increase in error which could be relevant here especially since we are measuring user engagement, whereas in RMSE this would be squared and hence higher which could be very relevant in critical applications like critical trials etc.
	- Model evaluation and final model:
		The trained model is evaluated based on the MAE on the test dataset. Out of the three models, the best performing model which is XGboost is saved using pickle.
	- Further improvements to the model:
		- We can try tuning other hyper parameters in the XGBoost model to see if the performance of the can be improved
		- Implement feature selection based on their importance to determine if some of them can be excluded from model training.
		- Try other regression based models like SVM regressor, Graidentboosting regressor, LightGBM regressor etc.. If the MAE observed is not acceptable in real situation, then may a deep learning regression model could improve the performance of the model
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