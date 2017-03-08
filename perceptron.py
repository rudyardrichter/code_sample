import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import sys


class Perceptron(object):

    """
    Class for perceptron classifers (simple 2-class model).
    """

    def __init__(self, d=None, log_file=None):
        """
        :d: int of dimension of the space
        :log: bool indicating whether to count mistakes made during training
        """
        self.d = d
        self.log_file = log_file
        if log_file is not None:
            self.take_log = True
            self.examples = 0
            self.mistakes = 0
            self.log = []
        else:
            self.take_log = False
        # Normal vector for the perceptron's boundary, initialized to 0.
        if d is not None:
            self.w = np.zeros(self.d)
        else:
            self.w = None

    def normalize_T(self, T):
        """
        Normalize a training data set row-wise.
        """
        return T / linalg.norm(T, axis=1).reshape(T.shape[0], 1)

    def train(self, T, U, batches):
        """
        Train the perceptron on the given data.
        :T: numpy.ndarray, n x d, consisting of training data
        :U: numpy.ndarray, n x d, consisting of training labels
        :batches: number of times to iterate over the training data
        :return: TODO
        """
        T = self.normalize_T(T)
        n = T.shape[0]
        # Initialize array of predictions for each training example.
        P = np.empty(U.shape, dtype=np.int8)
        # Run through the training set the number of times specified by
        # `batches`.
        for j in range(batches):
            # Run through all the points in the training set.
            for i in range(n):
                x_i = T[i]
                y_i = U[i]
                # Make a prediction based on the dot product of this data point
                # with the boundary normal vector.
                if np.dot(self.w, x_i) >= 0.:
                    P[i] = 1
                else:
                    P[i] = -1
                # Correct the normal vector if the prediction was wrong, and
                # also increment the mistakes if necessary.
                if P[i] != y_i:
                    self.w += y_i * x_i
                    if self.take_log:
                        self.mistakes += 1
                # Update the examples count and append to the log if the log is
                # being kept.
                if self.take_log:
                    self.examples += 1
                    self.log.append([self.examples, self.mistakes])
        # Save the log file if cumulative mistakes are being logged.
        if self.take_log:
            np.savetxt(self.log_file, self.log)
        return

    def train_from_files(self, T_file, U_file, batches):
        """
        Load files for training data and labels and train the perceptron on
        those.
        """
        T = np.loadtxt(T_file, dtype=np.float32)
        U = np.loadtxt(U_file, dtype=np.int8)
        if self.d is None:
            self.d = T[0].shape[0]
            self.w = np.zeros(self.d)
        self.train(T, U, batches)
        return

    def classify(self, A):
        """
        Classify the data A according to the current boundary of the
        perceptron.
        :A: numpy.ndarray, n x d
        :return: numpy.ndarray, n x 1
        """
        n = A.shape[0]
        # Initialize the array of classifications, defaulting (NOTE) to an
        # assignment of -1.
        C = np.full(n, -1, dtype=np.int8)
        # Take the row-wise dot product of the boundary normal vector with the
        # given data.
        dot_products = np.sum(A*self.w, axis=1)
        # Now, assign the appropriate entries with 1 where necessary. Note that
        # all the other entries already have been initialized to -1, as
        # desired.
        C[np.where(dot_products >= 0.)] = 1
        return C

    def classify_from_to_files(self, in_file, out_file):
        """
        Classify the points in A and save the results to the given file.
        """
        A = np.loadtxt(in_file, dtype=np.float32)
        C = self.classify(A)
        np.savetxt(out_file, C, fmt="%d")
        return


def main():
    train_file = sys.argv[1]
    label_file = sys.argv[2]
    test_file = sys.argv[3]
    predictions_file = sys.argv[4]
    batches = int(sys.argv[5])
    # Create a new perceptron instance.
    perceptron = Perceptron(log_file="mistakes.txt")
    # Train the perceptron on the data from the given files.
    perceptron.train_from_files(train_file, label_file, batches)
    # Classify the points from the test file, and save the results in the
    # predictions file.
    perceptron.classify_from_to_files(test_file, predictions_file)
    # Plot the cumulative mistakes from the perceptron's log.
    x, y = np.hsplit(np.array(perceptron.log), [1])
    plt.plot(x, y)
    plt.grid()
    plt.xlabel("Number of Examples Seen")
    plt.ylabel("Cumulative Number of Mistakes")
    plt.title("Cumulative Number of Mistakes in Perceptron Training")
    plt.savefig("perceptron_training_error.pdf")
    return


if __name__ == "__main__":
    main()
