import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
def displaying_prediction(numeric_result):
    if numeric_result[0] == 0:
        print("\nI predict if its space mission 1957")
    else:
        print("\n I predict if its space mission 1958")

# numeric_result

# Define the file
space = pd.read_csv(r"Sandibell_space_Mission part2  - Sheet1.csv")

# print the data sets space mission 1957
print("Is the space mission is 1957 or 1958")

# this the feature input toward 1957 and 1958 space mission
features = space[['Function', 'Operator', 'Status_Mission_Outcome']]

# this the Label input toward 1957 and 1958 space mission
labels = space[["Label_of_1957_or_1958"]]

# this classifier the feature and label input
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# print the data sets space mission 1957
print("\n 0 = space mission 1957, 1 = space mission 1958")
# 0	REV test	U.S Air Force	Successful
print("This example is from 1957. ")
print("Function=2, Operator=0, Status_Mission_Outcome=0")
print("")

# display the predict from 1957
result = clf.predict([['2', '0', '0']])
displaying_prediction(result)


# This display the predict from 1958
result = clf.predict([['13', '6', '0']])
displaying_prediction(result)


print("_" * 60)
print("\n\t*** Training Data into this application toward space mission trained on ***\n", space)
print("_" * 60)
# print the plot and this the size how I want to display
print(plt.figure(figsize=(20, 7)))
# I decide which one to print from features which is Status_Mission_Outcome to see the comparison
print(sns.countplot(x='Label_of_1957_or_1958', data=space))
plt.show()
# print the plot and this the size how I want to display
print(plt.figure(figsize=(16, 8)))

# I decide which one to print from features which is Operator to see the comparison
print(sns.countplot(x='Operator', data=space))
plt.show()

# This allow the user to input only space mission toward 1957 and 1958
# Also, ask question toward the user that was apart the mission
print("\n\t*** Analyze data set toward the space mission information ***\n")
Function = input("What function of the space mission where you?")
Operator = input("What operator of the space mission where you?")
Status_Mission_Outcome = input("What was the status mission outcome where you?")

# its display the prediction toward the question was answer will predict it which mission they where
result = clf.predict([[Function, Operator, Status_Mission_Outcome]])
displaying_prediction(result)

