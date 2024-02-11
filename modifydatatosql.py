def update_file(values):
    with open("data.txt", "w") as file:
        for value in values:
            file.write(str(value) + "\n")

def read_file():
    with open("data.txt", "r") as file:
        values = [str(line.strip()) for line in file.readlines()]
    return values

# Example usage
current_values = ['1.2, 5555454, 5.6,4545,67656456']

# Update the file with new values
update_file(current_values)

# Read values from the file
stored_values = read_file()

print("Stored Values:", stored_values)


