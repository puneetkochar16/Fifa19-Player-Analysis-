 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:08:26 2019

@author: Arpit Agarwal, Puneet Kochar & Avi Manawat
"""

#Importing packages that will be used in the project
import pandas as pd
import numpy as np
import datetime
import urllib3
import re
import seaborn as sns
import matplotlib.pyplot as plt


fifa19 = pd.read_csv('players_19.csv') #reads the fifa19 player dataset into python.


 
 
    
listofcolumn=[] #creates an empty list 

for j in fifa19: #loops through the fifa19 player dataset
    listofcolumn.append(j) #store the column name into the list

#loops into the columns of dataset    
for a in range(104):
    print(str(a)+' '+str(listofcolumn[a])) #returns the name of columns with index number.
 
#creates a new dataframe with selected columns we want to use for our analysis.
fifa19_1 = fifa19.iloc[:,[2,4,5,6,7,8,9,10,11,12,13,15,24,31,32,33,34,35,36,37,38,39,40,41,42]]

    
#Data Cleaning and Data Filtering.
a=fifa19_1[fifa19_1['team_position']=='RES'].index #looks for the Reserve players in dataset into the team position column.
fifa19_1.drop(a,inplace=True) #drops all the reserve players from dataset as we have no use of data of reserve player for analysis.
b=fifa19_1[fifa19_1['team_position']=='SUB'].index #looks for the Substitute players in dataset into the team position
fifa19_1.drop(b,inplace=True) #drops all the reserve players from dataset as we have no use of data of substitute player for analysis.

fifa19_1.dropna(subset=['team_position'], inplace=True) #drops all rows conatining Nan values based on team position.
fifa19_1['team_position'].unique()





fifa19_1['GK']=(fifa19_1['team_position']=='GK').astype(int)#converts categorical variable as indicator variable.


gkdata=fifa19_1.sort_values("GK",axis = 0, ascending = False).iloc[0:642,:] #creates a new dataframe called gkdata for all the goalkeepers in dataset.
otherplayer=fifa19_1.sort_values("GK",axis = 0, ascending = False).iloc[642:,:] #creates a new dataframe called otherplayer conatining data of all other players.
otherplayer.dropna(axis = 1, inplace=True) #drops all the rows contaning Nan values in otherplayer dataframe
gkdata.dropna(axis = 1, inplace=True)  #drops all the rows contaning Nan values in goalkeeper dataframe







#Regression Analysis with goalkeeper data.

"""
Goal: We want to predict the wage of goalkeeper based on overall rating and 
  various goalkeeping skills  of a goalkeeper
  
  """
"""
Hence, we are trying to create a model
   wage_euro = Beta0 + Beta1*overall + Beta2*gk_diving + Beta3*gk_handling + Beta4*gk_kicking
               +Beta5*gk_reflexes + Beta6*gk_speed
    """

  
#selects the column according to our dependent and predictable variables.
gkregdata = gkdata[['wage_eur','overall','gk_diving','gk_handling','gk_kicking','gk_reflexes','gk_speed']]


numberRows_gk = len(gkdata) #returns the length of gkdata and store it in a variable.
RandomlyShuffledRows = np.random.permutation(numberRows_gk)#randomly shuffling the rows of gkdata.

trainingRows_gk = RandomlyShuffledRows[0:515]#using first 515 rows for training
testRows_gk = RandomlyShuffledRows[515:]#remaining rows are test set

xTrain_gk = gkregdata.iloc[trainingRows_gk, 1:]#selecting training data with column index 1,2,3,4,and 5
yTrain_gk = gkregdata.iloc[trainingRows_gk, 0]#selecting training data of column index 0
xTest_gk = gkregdata.iloc[testRows_gk, 1:]#selecting test data with column index 1,2,3,4,5,6,7,8,and9
yTest_gk = gkregdata.iloc[testRows_gk, 0]#selecting test data of column index 0

from sklearn import linear_model 
regr = linear_model.LinearRegression() #create a linear regression model called regr
regr.fit(xTrain_gk,yTrain_gk) #fits training data into the model


wage_prediction = regr.predict(xTest_gk) 
errors = (wage_prediction-yTest_gk) #returs the error of Y prediction from actual y test data values.


print(regr.coef_) #prints value of Beta1,Beta2,Beta3,Beta4,Beta5,Beta6
print(regr.intercept_) #prints value of Beta0



#equation#
"""
 wage_euro = -180214.8233985338 + 3065.66202324*overall - 407.70987424*gk_diving + 21.47089081*gk_handling - 20.88955017*gk_kicking
               - 7.5609309Beta5*gk_reflexes + 164.2686999*gk_speed

""""



#other player Regression Anlaysis

"""
Goal: We want to predict the wage of other players based on overall rating and 
  various skills  of a player
  
"""
"""
Hence, we are trying to create a model
    wage_euro = Beta0 + Beta1*overall + Beta2*pace + Beta3*shooting + Beta4*passing
               +Beta5*dribbling + Beta6*defending
 """

#selects the column according to our dependent and predictable variables.
othplayerReg = otherplayer[['wage_eur','overall','pace','shooting','passing','dribbling','defending']]

numberRows_PL = len(otherplayer) #returns the length of gkdata and store it in a variable.
RandomlyShuffledRows = np.random.permutation(numberRows_PL)#randomly shuffling the rows

trainingRows_PL = RandomlyShuffledRows[0:5108]#using first 5108 rows for training
testRows_PL = RandomlyShuffledRows[5108:]#remaining rows are test set

xTrain_PL = othplayerReg.iloc[trainingRows_PL, 1:]#selecting training data with column index 1,2,3,4,5,and 6
yTrain_PL = othplayerReg.iloc[trainingRows_PL, 0]#selecting training data of column index 0
xTest_PL = othplayerReg.iloc[testRows_PL, 1:]#selecting test data with column index 1,2,3,4,5,6.
yTest_PL = othplayerReg.iloc[testRows_PL, 0]#selecting test data of column index 0

from sklearn import linear_model
regr1 = linear_model.LinearRegression() #create a linear regression model called regr1
regr1.fit(xTrain_PL,yTrain_PL)  #fits training data into the model

model_prediction = regr1.predict(xTest_PL)
diff = (model_prediction-yTest_PL) #returs the error of Y prediction from actual y test data values.


print(regr1.coef_) #prints value of Beta1,Beta2,Beta3,Beta4,Beta5,Beta6
print(regr1.intercept_)  #prints value of Beta0


from sklearn.metrics import mean_squared_error #imports mean_squared_error module from Sci-kit package.
mean_sqe = mean_squared_error(yTest_PL,model_prediction) #calculates average square error
print(mean_sqe)#prints average square error                                                   

from sklearn.metrics import r2_score#imports R-square module from Sci-kit Package.
r2 = r2_score(yTest_PL,model_prediction)#calculates the R-suared value
print(r2) #prints adjusted R-square.

p = 6 #number of predicted variables
n = len(testRows_PL) #length of test data set
adj_r2 = 1 - (1- r2)*(n-1)/(n-p-1) #calculates adjusted R-square
print(adj_r2) #prints adjusted R-square.

score=regr1.score(xTest_PL,yTest_PL)
print(score)

#equation

"""
 wage_euro =-209599.5170579833 + 3248.26602831*overall + 82.59527488*pace - 45.91949395*shooting - 45.11987663*passing
               +4.84415326*dribbling - 48.87999173*defending
"""


######## Log regression ###########


"""
since the data was exponenetially distributed so we applied log transformation
to bring uniformity

"""

trainingRows_PL = RandomlyShuffledRows[0:5108]#using first 320 rows for training
testRows_PL = RandomlyShuffledRows[5108:]#remaining rows are test set

xTrain_PL1 = othplayerReg.iloc[trainingRows_PL, 1:]#selecting training data with column index 1,2,3,4,5,and 6
yTrain_PL1 = np.log(othplayerReg.iloc[trainingRows_PL, 0])#selecting training data of column index 0
xTest_PL1 = othplayerReg.iloc[testRows_PL, 1:]#selecting test data with column index 1,2,3,4,5,6,7,8,and9
yTest_PL1 = np.log(othplayerReg.iloc[testRows_PL, 0])#selecting test data of column index 0

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(xTrain_PL1,yTrain_PL1)

model_prediction1 = reg.predict(xTest_PL1)#model_prediction is 1 or 0.
diff1 = (model_prediction1-yTest_PL1)


print(reg.coef_)
print(reg.intercept_)

score1=reg.score(xTest_PL1,yTest_PL1)
print(score1)




############# Descrpitive statistics of Various columns ###############################

fifa19_1["wage_eur"].describe()#returns mean, median, maximum, minimum, 1st quart., 3rd quart. and std.dev of wages of players.
fifa19_1["value_eur"].describe() #returns mean, median, maximum, minimum, 1st quart., 3rd quart. and std.dev of value of players.
fifa19_1["overall"].describe() #returns mean, median, maximum, minimum, 1st quart., 3rd quart. and std.dev of overall rating of players.
fifa19_1["potential"].describe() #returns mean, median, maximum, minimum, 1st quart., 3rd quart. and std.dev of potential of players.
fifa19_1["height_cm"].describe() #returns mean, median, maximum, minimum, 1st quart., 3rd quart. and std.dev of height of players.
fifa19_1["weight_kg"].describe()#returns mean, median, maximum, minimum, 1st quart., 3rd quart. and std.dev of weight of players.

#############line of best-fit between height and weight###############################

corr = fifa19_1["height_cm"].corr(fifa19_1["weight_kg"])

std_h =  fifa19_1["height_cm"].std()#standard deviation of Diameter
std_w = fifa19_1["weight_kg"].std()#standard deviation of Pigment

m = corr*(std_h/std_w)#slope of the line of best fit
xbar = fifa19_1["height_cm"].mean()
ybar = fifa19_1["weight_kg"].mean()

b = ybar-m*xbar

    
x=np.linspace(150,210,11)
plt.plot(fifa19_1["height_cm"],fifa19_1["weight_kg"],'ob')
plt.plot(x,m*x+b,'-ok')


############## Function ########################################

"""
function to calculate estimated wage of a player depending on overall rating and various skills.
"""
def player(overall,pace,shooting,passing,dribbling,defending):
    estimated_wage=regr.coef_[0]*overall+regr.coef_[1]*pace+regr.coef_[2]*shooting+regr.coef_[3]*passing+regr.coef_[4]*dribbling-regr.coef_[5]*defending+regr.intercept_
    print('Estimated Wage of Player :'+str(estimated_wage))
    return estimated_wage


overall1=float(input('Input Player overall rating ')) #takes input of overall rating from user.
pace1=float(input('Pace of player ')) #takes input of pace from user.
shooting1=float(input('Shooting Skills ')) #takes inpput of shootiong skills from user
passing1=float(input('Passing Skills ')) #takes input of passing skills from user.
dribbling1=float(input('Dribbling Skills ')) #takes input of Dribbling skills from user.
defending1=float(input('Defending Skills '))#takes input of defending skills from user.

player(overall1,pace1,shooting1,passing1,dribbling1,defending1) #configuration of skills of a player according to user input.





#####################Data Scraping##################
    
import urllib3 #imports the package to scrape the data from url
import re #imports regular expression package


urllib3.disable_warnings() #disables the warning

import pandas as pd
club=pd.read_csv('clubs.csv') #reads the club name file into python.

url = 'https://www.90min.com/posts/6244570-10-football-clubs-who-have-won-the-most-trophies-from-europe-s-top-5-leagues'
http = urllib3.PoolManager() 
response = http.request('GET', url) #stores all HTML code of webpage in variable response
webContent = str(response.data) #converts response to a string, called webContent



listofResults=[] #creates an empty list 
listpost=[] #creates an empty list to find position of number of cups won
#loops through club name column
for y in club['Club Name']:
    positionofClubName = webContent.find(y) #finds the location within webContent of the string "Club Name"
    newcontent = webContent[positionofClubName:positionofClubName+6000] #trims so we only consider characters upto 6000 position after "Previous CLose"
    resultofTrophies = re.findall("European Cup/Champions League x \d+",newcontent) #Search for strings within the file after Club Name
    listofResults.append(resultofTrophies) #append the list of results created with result values
print(listofResults) #prints list of Results

#since list contain single element. So we seprate all elements
list1=[]
for i in listofResults:
    for k in i:#take single element of list
        j=k
        list1.append(k)#append list in data

list2=[]        
for a in list1:
    value = re.sub("European Cup/Champions League x ","", a)#remove all the unwanted data
    no = int(value)#covnvert string to integer
    list2.append(no)#append the list
print(list2)

import pandas as pd
pd.DataFrame
club['Total Champions']=pd.DataFrame(list2)#create dataframe of list using pandas



club["Club Name"]=club["Club Name"].str.replace('(\d+)','')#remove all the number
club["Club Name"]=club["Club Name"].str.replace(')','')#removes one bracket so it will easier to find the location

    
    #to find total trophies
listpost1=[]
list5=[]
for q in club["Club Name"]:
    positionofTotal = webContent.find(q)#position of total trophies
    listpost1.append(positionofTotal)#to analyze position
    newcontent1 = webContent[positionofTotal:positionofTotal+25]#Find total trophies
    resultTotal = re.findall("\d\d",newcontent1)#to get total trophies
    list5.append(resultTotal)

print(list5)

#create single list into multiple list
list6=[]        
for m in list5:
    for w in m:
        no1 = int(w)
        list6.append(no1)
print(list6)

#save file to Datafram
pd.DataFrame
club['Total Trophies']=pd.DataFrame(list6)

club["Club Name"]=club["Club Name"].str.replace('(','')#changes club name to normal name


#plot bar graph with total trophies won and Champions league won by each club
club.plot.bar(x='Club Name')



 

"""
In accordance with proper visualization of data, 
we are taking a random sample data of players with sample size 500 from fifa19 dataset.

"""
noofRows = len(fifa19_1) #stores the length of dataset
random = np.random.permutation(noofRows) #=Randomly shuffling the rows of fifa19 dataset
sampleRows = random[0:500] #selction of first 500 rows from randomly shuffled rows.
sampleData = fifa19_1.iloc[sampleRows,0:500] #creating a sample datafram of our randomly shuffled rows.


import seaborn as sns #imports seaborn plotting package.


sns.lmplot(x='overall',y='wage_eur',data=sampleData) #plots linear model plot of Overall rating and wages of player
sns.lmplot(x='overall',y='wage_eur',data=sampleData,lowess=True) #plots best fir curve of Overall rating and wages of player

sampleData['Log']=np.log(sampleData['wage_eur'])
sns.lmplot(x='overall',y='Log',data=sampleData)
sns.lmplot(x='overall',y='Log',data=sampleData,lowess=True)
sns.boxplot(x='wage_eur',data=sampleData) #plots boxplot of distribution of wages of players

corrMatrix = sampleData.corr()
sns.heatmap(corrMatrix, annot = False) #plots correlation heatmaps of all the columns

sns.distplot(sampleData['wage_eur']) #plots histogram of wages column of players.


"""
Visualization of total wages of players among top 15 clubs

"""

listofmajorclubs=['Juventus','FC Barcelona','Real Madrid','Paris Saint-Germain','Manchester City','Manchester United',"Chelsea","Atlético Madrid","FC Bayern München","Tottenham Hotspur","Chelsea","Liverpool","Inter","Milan","Napoli","Arsenal"]

wage_per_team=[]  #creates an empty list. 
clb = fifa19_1.groupby("club") #creates category based on different clubs.

#loops into the list of major clubs to find the sum of wages all players in a particular club.
for q in listofmajorclubs:
    pi=clb.get_group(q)
    yi=pi['wage_eur'].sum() 
    wage_per_team.append(yi)
    
import matplotlib.pyplot as plt 
plt.bar(listofmajorclubs,wage_per_team) #plots bar chart of total wages of players among top 15 clubs
plt.xlabel("Clubs")
plt.ylabel("Total Player Wages")
plt.title("Total Wages of Players Among Top 15 Clubs")
plt.xticks(listofmajorclubs, fontsize=10, rotation=90)

#listofnationality=['Spain','Belgium','France','Germany','Argentina','Brazil']

"""
We created a function where by taking an input of the player name by user
It return the overall profile of the player

"""

listofplayers1=[]    
for playerName in fifa19.iloc[0:1000,2]:
    fullName=playerName.split()
    fullName.append(playerName)
    listofplayers1.append(fullName)
  

c=0 
y=1
name=input('Enter Player name ')
for individual in listofplayers1:
    for q in individual:
        if(name==q):
            c=y
    y=y+1

if(c>0):           
    print('Player Details are')
    removeEmpty=fifa19.iloc[c-1].dropna(axis=0, how='any')
    print(removeEmpty) 
else:
    print('No Player')
    
    
    
"""
Dream Team of fifa19 

"""
fifa19_1['team_position'] #selects column of team position
fifa19_2=fifa19_1.sort_values(by='team_position', ascending=False) #sort positions
fifa19_2['team_position']



listofpositions=[] #creates an empty list

#loops into sorted column of team position to get all positions in soccer.
for post in fifa19_2['team_position']:
    listofpositions.append(post)
unique=np.unique(listofpositions) #to get unique position

forString=[] #creates an empty list

#loops into th list of unique position to conver numpy string into string
for aa in unique:
    forString.append(str(aa))

bestplayer=fifa19_1.groupby('team_position') 
playerList=[] #creates an empty list

#loops through to create dataframes of different position
for zz in forString:
    xt=bestplayer.get_group(zz)  
    xt.sort_values(by='overall', ascending=False)
    bestplayerdata=str(xt.iloc[0,0])
    playerList.append(bestplayerdata)

#merging two list into one dataframe    
df={'Position':forString, 'Player Name':playerList}
dreamTeam=pd.DataFrame(df)
print(dreamTeam)


##################K-means Clusttering##################



import pandas as pd


fifa19["date"] = pd.to_datetime(fifa19["dob"])  #treats column like dates/time


import datetime 
listofdays=[] #creates an empty list 
#loops into the fifa19 dataset
#for vv in range(len(fifa19_1)):
#    noofdays = (datetime.datetime.today()-fifa19_1["date"][i]).days #calculates the current age of player in number of days.
#    listofdays.append(noofdays) #stores value of number of days into listofdays.
    
for vv in fifa19["date"]:
    noofdays = (datetime.datetime.today()-vv).days #calculates the current age of player in number of days.
    listofdays.append(noofdays) #stores value of number of days into listofdays.
    
fifa19['days']=listofdays

from sklearn import cluster #imports cluster module from Sci-Kit package.

    
cats = ["days","value_eur","wage_eur"] #selects category of columns we want to deal with

selectedData = fifa19.loc[:,cats] #creates a new dataframe with selected columns

from sklearn.preprocessing import StandardScaler #imports scaling module from Sci-kit package
scaler = StandardScaler() #scales the data
scaledData = scaler.fit_transform(selectedData) #fits the scaling data



k = 10 #number of cluster to be formed.

kMeansResult = cluster.KMeans(k).fit(scaledData) #fits the data around scaled data using k-means
kMeansResult.labels_ #gives labeleling to each data point
labelSymbols = ["*","+","o","s","^","x",".","p",">","4"] #give a different symbol for each cluster
labelColors = ['r','b','k','g','c','m','y','cyan','grey','navy'] #give a different color for each cluster




import matplotlib.pyplot as plt

#loops into the length of selected data to form clusters of age(in days) and value of player in euro.
for i in range(len(selectedData)):
    groupNumber = kMeansResult.labels_[i]
    symbol = labelSymbols[groupNumber]
    col = labelColors[groupNumber]
    plt.scatter(selectedData.loc[i,"days"],selectedData.loc[i,"value_eur"],marker=symbol,c=col)
    plt.xlabel('Age')
    plt.ylabel('Value')
    
#loops into the length of selected data to form clusters of age(in days) and wage of players.
for i in range(len(selectedData)):
    groupNumber = kMeansResult.labels_[i]
    symbol = labelSymbols[groupNumber]
    col = labelColors[groupNumber]
    plt.scatter(selectedData.loc[i,"days"],selectedData.loc[i,"wage_eur"],marker=symbol,c=col)
    plt.xlabel('Age')
    plt.ylabel('Wage')













