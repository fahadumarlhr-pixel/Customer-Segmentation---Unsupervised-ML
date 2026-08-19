import pandas as pd

# Loadind the Dataset :- 
df = pd.read_csv("Mall_Customers.csv")

# Exploring :-
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# Input
Input_features = [
    "Age",
    "Annual_Income",
    "Spending_Score"
]



