import json

# Put the path to one JSON file
json_file = r"C:\Users\prafful\Desktop\dataset\ann_train_val\peace.json"

# Open JSON
with open(json_file, "r") as f:
    data = json.load(f)

# Total samples in file
print("Total Samples:", len(data))

# Get first sample
first_key = next(iter(data))

print("\nFirst Sample ID:")
print(first_key)

print("\nAvailable Fields:")
print(data[first_key].keys())

print("\nLabel:")
print(data[first_key]["labels"])

print("\nNumber of Landmarks:")
print(len(data[first_key]["landmarks"][0]))

print("\nFirst Landmark:")
print(data[first_key]["landmarks"][0][0])