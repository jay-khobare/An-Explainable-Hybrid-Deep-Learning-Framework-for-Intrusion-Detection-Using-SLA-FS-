# Data Merging
import pandas as pd

df2 = pd.read_csv("MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv")
df4 = pd.read_csv("MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df1 = pd.read_csv("MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df4.columns = df4.columns.str.strip()

df = pd.concat([df1, df2, df4], axis=0, ignore_index=True)

print("Total Shape:", df.shape)
print(df['Label'].value_counts())

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("Merged Data/final_merged_dataset.csv", index=False) 