import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    

    return model, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(["Unnamed: 32", "id"], axis=1)

    # we need Malignant (M) in diagnosis as 1 cause it is the target to detect malicious tumor or not
    # for that we use map function to convert M to 1 and B to 0
    data['diagnosis'] = data['diagnosis'].map({ 'M' : 1, 'B' : 0 })

    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    # data = pd.read_csv("data/data.csv")
    # print(data.info())

    # build the model apart, export it to binary file, and then import the binary file to application
    # because building the training and test in the same creates more problem to usability cause we have to run 
    # the ml model always when logging into the application so we use pickle5 module for this reason
    with open('model/model.pkl', 'wb') as f:
        # opening a new pickle file called model.pkl and then writing into file so 'w'
        # and its a binary file so 'b' --> 'wb' as the second argument and f for file
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()