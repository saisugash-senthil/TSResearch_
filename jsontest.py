import json

# Load existing JSON data from the file
with open('model_namesupt.json', 'r') as file:
    data = json.load(file)

# Add new elements to the list
new_elements = [
    {"model_202406hng210_080000.pkl": None},

]

data.extend(new_elements)

# Save the updated data back to the file
with open('model_namesupt.json', 'w') as file:
    json.dump(data, file, indent=2)

print("Elements added successfully.")
