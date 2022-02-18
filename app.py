from flask import Flask , render_template , request
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='ANN_churn.h5'

# Load your trained model
model = load_model(MODEL_PATH)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    gender_female = 0
    geography_germany=0
    geography_spain=0

    if request.method == 'POST':
        gender_male=request.form['gender_male']
        if(gender_male=='male'):
            gender_male=1
            gender_female=0
        else:
            gender_male=0
            gender_female=1
            
        age = int(request.form['age'])
        geography_france=request.form['geography_france']
        if(geography_france=='france'):
            geography_france=1
            geography_germany=0
            geography_spain=0
        elif(geography_germany=='germany'):
            geography_france=0
            geography_germany=1
            geography_spain=0
        else:
            geography_france=0
            geography_germany=0
            geography_spain=1
        
        hascrcard_yes=request.form['hascrcard_yes']
        if(hascrcard_yes=='yes'):
            hascrcard_yes=1
        else:
            hascrcard_yes=0

        isactive_yes=request.form['isactive_yes']
        if(isactive_yes=='yes'):
            isactive_yes=1
        else:
            isactive_yes=0
        
        creditscore = int(request.form['creditscore'])

        tenure = int(request.form['tenure'])

        balance = float(request.form['balance'])

        numofproduct = int(request.form['numofproduct'])

        estimatedsal = float(request.form['estimatedsal'])
            
            
        input_data = ([[creditscore,age,tenure,balance,numofproduct,hascrcard_yes,isactive_yes,estimatedsal,geography_france,geography_germany,geography_spain,gender_female,gender_male]])

        input_data_numpy_array = np.asarray(input_data)

        reshape_input_data = input_data_numpy_array.reshape(1,-1)

        prediction = model.predict(reshape_input_data)
        
        if(prediction[0]==0):
            return render_template('result.html',prediction_texts="Customer will not churn!!!")
        else:
            return render_template('result.html',prediction_texts="Customer will churn!!!")
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)