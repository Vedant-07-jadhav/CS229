import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    accuracy = np.mean((y_pred > 0.5) == y_eval)
    print(f'Validation accuracy: {accuracy:.4f}')

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        phi = np.mean(y)
        mu_0 = np.mean(x[y == 0], axis=0)
        mu_1 = np.mean(x[y == 1], axis=0)

        x_0 = x[y == 0] - mu_0
        x_1 = x[y == 1] - mu_1
        sigma = (x_0.T @ x_0 + x_1.T @ x_1) / m

        sigma_inv = np.linalg.inv(sigma)

        self.theta[1:] = sigma_inv @ (mu_1 - mu_0)
        self.theta[0] = -0.5 * mu_0 @ sigma_inv @ mu_0 + 0.5 * mu_1 @ sigma_inv @ mu_1 + np.log(phi / (1 - phi))

        return self.theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE

main(train_path='/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/data/ds1_train.csv', eval_path='/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/data/ds1_valid.csv',pred_path='/home/vedant/Codes&Projects/CS229/problem_sets/PS_1/output/pred_ques3/p01e_pred.txt')