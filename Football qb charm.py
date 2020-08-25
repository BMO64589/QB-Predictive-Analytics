import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from models import *

################## Load and Process Data ###############################################################
ds_data = pd.read_csv(r'E:\Sports Data\ASA NFL All Raw Data.csv')

ds_data['Week'] = ds_data['Week'].astype('str')
ds_data['Year'] = ds_data['Year'].astype('str')

fb1_data = ds_data[['Player', 'Pos', 'Year', 'Week', 'Team', 'Opponent Abbrev',
                    'Date', 'H/A', 'Team Score', 'Opponent Score', 'Actual Margin', 'Indoor/Outdoor',
                    'Surface', 'Weather', 'Temperature', 'Wind', 'Pass Att', 'Pass Cmp', 'Pass Yds',
                    'Pass TD', 'Pass Int', 'Pass Lng', 'Pass Rate', 'Pass Sk', 'Pass Sk Yds',
                    'Rush Att', 'Rush Yds', 'Rush TD', 'Rush Lng']]

pd.set_option('display.max_columns', 500)
display(fb1_data)

AR_DP = fb1_data[fb1_data['Player'].isin(['Dak Prescott', 'Aaron Rodgers'])]
players = ['Dak Prescott', 'Aaron Rodgers']
colors = mcolors.BASE_COLORS
tableau_colors = mcolors.TABLEAU_COLORS
colors.update(tableau_colors)
colors = list(tableau_colors.keys())

################################ Plot Pass yards by Game ####################################################
for player in players:
    plt.figure(figsize=(20, 20))
    for i in range(6, 10):
        years = '201{}'.format(i)
        year_data = AR_DP[(AR_DP.Year == years) & (AR_DP.Player == player)]
        x_data = [i for i in range(1, 18)]
        y_data = [year_data[year_data.Week == str(x)]['Pass Yds'].values[0] if str(x) in year_data.Week.values else 0
                  for x in x_data]
        labels = [year_data[year_data.Week == str(x)]['Opponent Abbrev'].values[0] if str(
            x) in year_data.Week.values else 'N/A' for x in x_data]
        plt.scatter(x_data, y_data, c=colors[i], label=years)
        plt.xlabel('Week Number')
        plt.xticks(np.arange(min(x_data), max(x_data) + 1, 1.0))
        plt.ylabel('Pass Yards')
        plt.yticks(np.arange(min(y_data), max(y_data) + 1, 10.0))
        plt.legend()
        plt.title('Overview {}'.format(player))
        for i, label in enumerate(labels):
            plt.annotate(label, (x_data[i], y_data[i]))
    ##plt.savefig(os.path.join('E:\Python Scripts','passingyards_{}.{}.png'.format(player.split(' ')[0][0], player.split(' ')[1])))
    plt.show()
    plt.clf()


############### Multi regression ################################################################

## multiple regression
def mult_reg(x, y):
    reg = LinearRegression().fit(x, y)  # To get coefficients
    coeffs = reg.coef_
    # To get intercept
    b = reg.intercept_
    # To evaluate performance
    score = reg.score
    return coeffs, b, score


QB = fb1_data[fb1_data['Pos'].isin(['QB'])]

y = QB['Pass Yds'].values
x1 = QB['Opponent Score'].values
x2 = QB['Actual Margin'].values

x = np.stack((x1, x2)).T

coeffs, b, score = mult_reg(x, y)
print(coeffs, b, score)

model = sm.OLS(y, x)
res = model.fit()

print(res.summary())

################### manipulating data ##########################################################
fb1_data['W/L'] = (np.sign(fb1_data['Team Score'] - fb1_data['Opponent Score']) + 1) / 2
print(fb1_data)

###################################### K means #################################################

QB = fb1_data[fb1_data['Pos'].isin(['QB'])]
X = QB[['Team', 'Team Score', 'Indoor/Outdoor', 'Surface',
        'Temperature', 'Wind', 'Pass Att', 'Pass Cmp', 'Pass Yds',
        'Pass TD', 'Pass Int', 'Pass Lng', 'Pass Rate', 'Pass Sk', 'Pass Sk Yds']]
Y = QB[['Player']].values
X['Team'] = pd.factorize(X['Team'])[0]
X['Indoor/Outdoor'] = pd.factorize(X['Indoor/Outdoor'])[0]
X['Surface'] = pd.factorize(X['Surface'])[0]

X[X.isnull().any(axis=1)]
replace_indices = X.isnull().any(axis=1)
X.Temperature[replace_indices] = 64.0
X.Wind[replace_indices] = 0.0

X = X.values
scaler = StandardScaler()  # Data Scaler object to standard normal dist

X_norm = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_norm)
labels = kmeans.labels_

pca = PCA(n_components=2)
Z = pca.fit_transform(X_norm)

tableau_colors = mcolors.TABLEAU_COLORS
colors = list(tableau_colors.keys())

for i in range(max(labels) + 1):
    cluster_data = Z[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[i], label='Cluster: {}'.format(i))

plt.legend()
plt.show()
plt.clf()

################################## Hist bar chart ##############################################
for i in range(max(labels) + 1):
    unique, counts = np.unique(Y[labels == i], return_counts=True)
    counts_dict = dict(zip(unique, counts))
    sorted_count = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)[:10]  # Take first ten items from list
    top_ten_players = [pair[0] for pair in sorted_count]  # Extract player labels (x)
    top_ten_frequencies = [pair[1] for pair in sorted_count]  # Extract frequencies
    plt.bar(top_ten_players, top_ten_frequencies)  # Create a bar graph
    plt.tick_params(axis='x', width=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.clf()

##############Using a counter:

##count_dict = dict(Counter(Y[labels == i]))  # Count the frequencies within each cluster
##sorted = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:10]  # Take first ten items from list

##top_ten_players = [pair[0] for pair in sorted]  # Extract player labels (x)
##top_ten_frequencies = [pair[1] for pair in sorted]  # Extract frequencies

##plt.bar(top_ten_players, top_ten_frequencies)  # Create a bar graph

############################# Passing Yards Perdicter + One Hot Vector ############################################

X_R = QB[['Team', 'Team Score', 'Indoor/Outdoor', 'Surface',
          'Temperature', 'Wind', 'Pass Att', 'Pass Cmp', 'Player',
          'Pass TD', 'Pass Int', 'Pass Lng', 'Pass Rate', 'Pass Sk', 'Pass Sk Yds']]
Y_R = QB[['Pass Yds']].values
X_R['Team'] = pd.factorize(X_R['Team'])[0]
X_R.drop(columns=['Team'])
X_R['Indoor/Outdoor'] = pd.factorize(X_R['Indoor/Outdoor'])[0]
X_R['Surface'] = pd.factorize(X_R['Surface'])[0]
X_R['Player'] = pd.factorize(X_R['Player'])[0]
One_Hot_Team = OneHotEncoder().fit_transform(X_R['Team'].values.reshape(-1, 1)).toarray()
One_Hot_IO = OneHotEncoder().fit_transform(X_R['Indoor/Outdoor'].values.reshape(-1, 1)).toarray()
One_Hot_Player = OneHotEncoder().fit_transform(X_R['Player'].values.reshape(-1, 1)).toarray()
One_Hot_Surface = OneHotEncoder().fit_transform(X_R['Surface'].values.reshape(-1, 1)).toarray()
X_R.drop(columns=['Team', 'Player', 'Surface', 'Indoor/Outdoor'])

X_R[X_R.isnull().any(axis=1)]
replace_indices = X_R.isnull().any(axis=1)
X_R.Temperature[replace_indices] = 64.0
X_R.Wind[replace_indices] = 0.0

X_R = X_R.values
scaler = StandardScaler()  # Data Scaler object to standard normal dist
X_R = np.hstack((X_R, One_Hot_Team, One_Hot_IO, One_Hot_Player, One_Hot_Surface))
X_norm_R = scaler.fit_transform(X_R)
##mult_reg(X_norm_R, Y_R)
print(X_norm_R)
print(X_R[0, :])

###################### Nerual Network Model####################################
XR_train, XR_test, YR_train, YR_test = train_test_split(X_norm_R,  Y_R, test_size=0.2, random_state=42)
XR_train_R, XR_val, YR_train_R, YR_val = train_test_split(XR_train,  YR_train, test_size=0.2, random_state=42)


############################# Hyper Param ###########################################################

lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1]
num_layers = [i for i in range(2,10)]
best_model, results = grid_hyperparamsearch(lrs, num_layers, XR_train_R, YR_train_R, XR_val, YR_val, XR_test, YR_test, numepochs=10)
############################## Object Oriented ###############################################

Model = MyModel()
print (Model.call(np.random.normal(size = (1000, 1))))





    

