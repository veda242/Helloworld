import pandas as pd
dataset=pd.read_csv(r"C:\Users\NavyaVeda\Downloads\fitness_tracker_dataset.csv (1)\fitness_tracker_dataset.csv")
print(dataset.head(10))
dataset.fillna(dataset.mean, inplace=True)
i=dataset["steps","distance_km","active_minutes"]