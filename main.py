import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from helpers import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, confusion_matrix, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train.reshape((-1, 1)), y_train.reshape((-1, 1)))
    print("Score: ", reg.score(X_train.reshape((-1, 1)), y_train.reshape((-1, 1))))
    print("Coefficients: ", reg.coef_)
    print("Intercept: ", reg.intercept_)

    y_pred = reg.predict(X_test.reshape((-1, 1)))

    model = sm.OLS(y_train, X_train).fit()
    slope = model.params[0]
    n = len(X_train)
    p = 1  # number of parameters
    t = slope / model.bse[0]
    r2 = r2_score(y_test, y_pred)

    # print the t-statistic
    print("t-statistics: ", t)
    print("R2 score: ", r2)

    clf_x = df['views'].values
    clf_y = df['sentiment'].values

    clf_y = label_binarize(clf_y, classes=[0, 1, 2])
    n_classes = 3
    clf_x_train, clf_x_test, clf_y_train, clf_y_test = train_test_split(clf_x, clf_y, test_size=0.2)
    clf = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000))
    y_score = clf.fit(clf_x_train.reshape((-1, 1)), clf_y_train.ravel()).decision_function(clf_x_test)

    print("Logistic Regression params: ", clf.coef_, clf.intercept_)
    print("Classifier Score: ", clf.score(clf_x_train.reshape((-1, 1)), clf_y_train.ravel()))
    clf_y_pred = clf.predict(clf_x_test.reshape((-1, 1)))
    print("Pseudo r squared: ", efron_rsquare(clf_y_test, clf_y_pred))

    print(confusion_matrix(clf_y_test, clf_y_pred))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(clf_y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(clf_y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
