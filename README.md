# Apps_predicions
How app characteristics can affect the success of an app

<h2><b>Abstract</b></h2>



   The following research explores how can multiple mobile app characteristics affect the way the users
view the app. Likewise, it tries to find the best accuracy for the classification models that predict if an
app will be successful or not based on a multitude of app characteristics.

   The user rating will be considered as an indicator for the success of an app. In this research paper the
apps which have a rating under 4 have been considered not successful while those of 4 and above have
be considered successful.

   The dataset was obtained from Kaggle, but the creator of the dataset extracted it from the iTunes
Search API which can be found on the Apple website. Linux web scraping tools were used to get the
dataset. The classification methods used to find the best accuracy were Random Forest, Light Gradient
Bosting Machine and Multilayer Perceptron. The best hyperparameters for those methods were find
with the help of Grid Search.

  In this research it was found the number of languages that the app is available on, the number of images
on the iOS Store page of the app and the length of the description might have an impact on the user
rating. Likewise, the best accuracy of classification methods was around 68% and it was obtained by
training the Multilayer Perceptron.

   This means that making sure that your app is available in multiple languages and that you have a good
presentation of the app on the iOS Store is quite important for having a successful app and maybe one
that can arrive in the tops.


<h2><b>Introduction</b></h2>


  The mobile app industry is frequently changing and this makes developing and maintaining an app
competitive a difficult task. The number of mobile appplications it is getting a higher and higher and it
has an edge over desktop applications. iOS has 43% of the smartphone market while Android has about
53%. If your application cannot be found easily, it has bugs or its presentation on the app store is not
professional, the number of downloads and user rating will be low. This means that mobile app
analytics are very important for the growth and retention of the customer base. 

<h3><b>Dataset</h3></b>

The dataset was obtained from <a href="https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps">Kaggle</a>.

<h3><b> Specifications </h3></b>

In this project Hyperopt was used to find the best hyperparamets for the classifiers.
