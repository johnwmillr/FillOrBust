import time
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

from dice.agents import Smart


# Load the trained agent
print("Loading the trained Agent.")
fp = "./dice/trained_agents/trained_agent_1710054236_25000.pkl"
with open(fp, "rb") as fin:
    agent: Smart = pickle.load(fin)

# Format the data for the model
print("Encoding the training data.")
encoder = OneHotEncoder(sparse_output=True, handle_unknown="infrequent_if_exist")
rows: list[tuple[str, ...]] = []
for state, action in agent.X:
    rows.append(tuple(state.split("+")) + (action,))
features = pd.DataFrame(
    rows,
    columns=["card", "discard", "banked_score", "turn_score", "dice", "holds", "action"],
).tail(2_500_000)

# Encode the categorical features
numerical_columns = ["banked_score", "turn_score"]
numerical_features = features[numerical_columns]
for nc in numerical_columns:
    numerical_features = (
        numerical_features
        .assign(**{nc: lambda df: df[nc].str.split("_").str.get(-1).astype(int)})
    )
categorical_features = features.drop(columns=numerical_columns)
encoded_categorical_features = encoder.fit_transform(categorical_features)

# Recombine the numerical and categorical features
X = hstack((numerical_features, encoded_categorical_features))
y = pd.Series(agent.y).iloc[features.index]

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=31
)

# Train a model to estimate the reward for taking a particular action from a given state
print("Training the estimator.")
estimator = RandomForestRegressor(verbose=1, n_jobs=-1)
estimator.fit(X_train, y_train)

# Evaluate the model
y_pred = estimator.predict(X_test)
print(f"{mean_squared_error(y_test, y_pred)=:0.3f}")
print(f"{r2_score(y_test, y_pred)=:0.3f}")

# Save the trained encoder and model
save_time = int(time.time())
print("Saving the trained encoder and estimator.")
for name, item in zip(["encoder", "estimator"], [encoder, estimator]):
    fp = f"./dice/dice_trained_{name}_{str(estimator.__class__).rsplit('.', 1)[1][:-2]}_{save_time}.pkl"
    print(fp)
    with open(fp, "wb") as fout:
        pickle.dump(item, fout)
print("Done!")
