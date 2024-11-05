import pandas as pd
dataset = pd.read_csv(r"C:\Users\NavyaVeda\Downloads\fitness_tracker_dataset.csv (2)\fitness_tracker_dataset.csv")
dataset.fillna(dataset.mean(),inplace=True)
input_data = dataset[['steps','distance_km','workout_type','sleep_hours','weather_conditions','active_minutes']]
output_data = dataset[['calories_burned','mood']]

from sklearn.preprocessing import LabelEncoder
label_encoder_workout = LabelEncoder()
label_encoder_weather = LabelEncoder()
dataset['workout_type'] = label_encoder_workout.fit_transform(dataset['workout_type'])
dataset['weather_conditions'] = label_encoder_workout.fit_transform(dataset['weather_conditions'])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
input_data_scaled=scaler.fit_transform(input_data)

from sklearn.model_selection import train_test_split
input_data_train,input_data_test,output_data_train,output_data_test=train_test_split(input_data_scaled,output_data,test_size=0.2,random_state=42)

from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense #type:ignore

#desigining the model
model = Sequential()
model.add(Dense(64, input_dim=6, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear'))  


model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
history = model.fit(input_data_train,output_data_train,validation_data=(input_data_test,output_data_test),epochs=100,batch_size = 32)

loss,mae = model.evaluate(input_data_test,output_data_test)
print(f'Test Loss:{loss},Test MAE:{mae}')

from sklearn.metrics import mean_squared_error
import numpy as np
output_data_pred = model.predict(input_data_test)
rmse = np.sqrt(mean_squared_error(output_data_test,output_data_pred))
print(f'RMSE:{rmse}')

import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label = 'train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

from tensorflow.keras.layers import Dropout #type:ignore
model.add(Dropout(0.5))

model.save('fitness_tracker_ann.h5')

from fastapi import FastAPI, Request
from tensorflow.keras.models import load_model #type:ignore
import numpy as np
from pydantic import BaseModel

model = load_model('fitness_tracker_ann.h5')
app = FastAPI()

class InputData(BaseModel):
    steps : int
    distance_km : float
    workout_type : str
    sleep_hours : float
    weather_conditions : str
    active_minutes : float

@app.post("/predict")
async def predict(data:InputData):
    workout_type_encoded = label_encoder_workout.transform([data.workout_type])[0]
    weather_conditions_encoded = label_encoder_weather.transform([data.weather_conditions])[0]

    i_data = np.array([[data.steps,data.distance_km,data.workout_type,data.sleep_hours,data.weather_conditions,data.active_minutes]])
    i_data_scaled = scaler.transform(i_data)

    predicton = model.predict(i_data)
    return{"calories_burned":predicton[0][0],"mood":predicton[0][1]}