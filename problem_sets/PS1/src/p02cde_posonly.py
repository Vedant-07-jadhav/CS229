import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    # c Part
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept= True)
    model_t = LogisticRegression()
    model_t.fit(x_train, t_train)

    x_test, t_test = util.load_dataset(test_path, label_col='t',add_intercept=True)
    util.plot(x_test, t_test, model_t.theta, '/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/output/p02c.png')

    t_pred = model_t.predict(x_test)
    np.savetxt(pred_path_c, t_pred > 0.5, fmt='%d')

    # d part
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    model_y = LogisticRegression()
    model_y.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    util.plot(x_test, y_test, model_y.theta, '/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/output/p02d.png')

    y_pred = model_y.predict(x_test)
    np.savetxt(pred_path_d, y_pred> 0.5, fmt='%d')

    # e part
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    alpha = np.mean(model_y.predict(x_val))
    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, '/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/output/p02e.png', correction)

    t_pred_e = y_pred / alpha
    np.savetxt(pred_path_e, t_pred_e > 0.5, fmt='%d')
    # *** END CODER HERE

main('/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/data/ds3_test.csv', '/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/data/ds3_valid.csv', '/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/data/ds3_train.csv', '/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/output/pred_Ques_2/p02X_pred.txt')
