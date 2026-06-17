import joblib

model = joblib.load("champion_model.pkl")

print(type(model))
print(model.hidden_layer_sizes)
print(model.n_iter_)
print(model.out_activation_)