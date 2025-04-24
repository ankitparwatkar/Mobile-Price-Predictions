import numpy as np
import pandas as pd
import pickle
from flask import  Flask,render_template,request
app = Flask(__name__, static_url_path='/static')
df=pd.read_csv("dataset.csv")
x=df.iloc[:,:20]
y=df['price_range']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=30, n_estimators=200)
x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)
model.fit(x_train_df, y_train_df)
with open ('mobile.pkl','wb') as file:
    pickle.dump(model,file)
from sklearn.metrics import accuracy_score, confusion_matrix
x_test_df = pd.DataFrame(x_test)
y_pred = model.predict(x_test_df)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)
print(confusion_matrix(y_test, y_pred))
with open ('mobile.pkl','rb') as file:
    model1 = pickle.load(file)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(x_train_df, y_train_df)
print("Best Parameters:", grid_search.best_params_)
import matplotlib.pyplot as plt
import seaborn as sns
plt.plot(y_test, y_pred, 'o', color='black')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color='red', linewidth=1.5)
plt.show()
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()
new_data = pd.DataFrame({
    'battery_power': [842],
    'blue': [0],
    'clock_speed': [2.2],
    'dual_sim': [1],
    'fc': [1],
    'four_g': [1],
    'int_memory': [53],
    'm_dep': [0.8],
    'mobile_wt': [136],
    'n_cores': [5],
    'pc': [6],
    'px_height': [905],
    'px_width': [756],
    'ram': [2631],
    'sc_h': [16],
    'sc_w': [7],
    'talk_time': [19],
    'three_g': [0],
    'touch_screen': [1],
    'wifi' : [0]
})
new_predictions = model.predict(new_data)
print("Predicted target value(s):", new_predictions)
print("0=lowcost: 10000-20000 | 1=mediumcost: 20500-35000 | 2=highcost: 35500-70000 | 3=veryhighcost: 70000+")
@app.route('/')
def Home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
     battery_power=float(request.form['battery_power'])
     blue = float(request.form['blue'])
     clock_speed = float(request.form['clock_speed'])
     dual_sim = float(request.form['dual_sim'])
     fc = float(request.form['fc'])
     four_g = float(request.form['four_g'])
     int_memory = float(request.form['int_memory'])
     m_dep = float(request.form['m_dep'])
     mobile_wt = float(request.form['mobile_wt'])
     n_cores = float(request.form['n_cores'])
     pc = float(request.form['pc'])
     px_height = float(request.form['px_height'])
     px_width = float(request.form['px_width'])
     ram = float(request.form['ram'])
     sc_h = float(request.form['sc_h'])
     sc_w = float(request.form['sc_w'])
     talktime = float(request.form['talktime'])
     three_g = float(request.form['three_g'])
     touch_screen = float(request.form['touch_screen'])
     wifi = float(request.form['wifi'])
     
     result= model.predict([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talktime, three_g, touch_screen, wifi]])[0]
     return render_template('index.html',result="{}".format(result))
if __name__=="__main__":
    app.run(debug=True)