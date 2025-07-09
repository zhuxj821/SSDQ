import pandas as pd

# File path
file_path = "SPEAKERS.TXT"

# Read file
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Extract ID, SEX, and SUBSET
id_sex_mapping = []

for line in lines:
    parts = line.strip().split("|")
    if len(parts) >= 4:
        speaker_id = parts[0].strip()
        sex = parts[1].strip()
        subset = parts[2].strip()
       # if subset == "train-clean-100":
        id_sex_mapping.append((speaker_id, sex, "English"))

# Convert to DataFrame
df = pd.DataFrame(id_sex_mapping, columns=["ID", "SEX", "LANG"])

# Save to CSV
csv_file_path = "speaker.csv"
df.to_csv(csv_file_path, index=False)

# Display first few rows
print(df.head())
