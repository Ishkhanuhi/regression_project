import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from helpers import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix
from sklearn import tree


def main():
    # df = generate_data()
    # write_csv(df)
    # correct_descriptions()
    # df = pd.read_csv("cleaned_data.csv")
    # tf_idf_df = calculate_tf_idf(df['description'])
    # save_tf_idf(tf_idf_df)
    # save_sentiments()
    df = pd.read_csv("data_with_sentiment_analysis.csv", index_col=False)

    X = df['views'].values
    y = df['likes'].values
    plt.scatter(X, y)
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train.reshape((-1, 1)), y_train.reshape((-1, 1)))
    print("Score: ", reg.score(X_train.reshape((-1, 1)), y_train.reshape((-1, 1))))
    print("Coefficients: ", reg.coef_)
    print("Intercept: ", reg.intercept_)
    plt.plot(X, reg.coef_[0] * X + reg.intercept_, color='r')
    plt.show()
    y_pred = reg.predict(X_test.reshape((-1, 1)))

    model = sm.OLS(y_train, X_train).fit()
    slope = model.params[0]
    t = slope / model.bse[0]
    r2 = r2_score(y_test, y_pred)

    # print the t-statistic
    print("t-statistics: ", t)
    print("R2 score: ", r2)

    clf_x = df['views'].values
    clf_y = df['sentiment'].values

    plt.bar(clf_y, clf_x)
    plt.show()

    le = LabelEncoder()
    le.fit(clf_y)

    clf_y = le.transform(clf_y)
    clf_x_train, clf_x_test, clf_y_train, clf_y_test = train_test_split(clf_x, clf_y, test_size=0.2)
    log_model = LogisticRegression()
    log_model.fit(clf_x_train.reshape((-1, 1)), clf_y_train)
    clf_y_pred = log_model.predict(clf_x_test.reshape((-1, 1)))
    print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(
        log_model.score(clf_x_test.reshape((-1, 1)), clf_y_test)))
    print(confusion_matrix(clf_y_test, clf_y_pred))

    clf = tree.DecisionTreeClassifier(max_depth=20)
    clf.fit(clf_x_train.reshape((-1, 1)), clf_y_train)
    y_pred = clf.predict(clf_x_test.reshape((-1, 1)))
    correct_predictions = (y_pred == clf_y_test).sum()
    accuracy = correct_predictions / len(y_test)
    print("Accuracy:", accuracy)
    r = tree.export_text(clf, feature_names=['views'])
    print(r)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
