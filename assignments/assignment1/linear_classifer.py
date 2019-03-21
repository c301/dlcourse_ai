import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    is_batch = len(predictions.shape) > 1
    if is_batch:
        max = np.max(predictions, axis=1).reshape((-1,1))
    else:
        max = np.max(predictions)

    exp = predictions - max
    exp = np.exp(exp)

    if is_batch:
        exp_sum = np.sum(exp, axis=1).reshape((-1,1))
    else:
        exp_sum = np.sum(exp)

    return exp / exp_sum


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    is_batch = len(probs.shape) > 1

    if not is_batch:
        q_x = probs[target_index]
        if q_x > 0:
            return -np.log(q_x)
        else:
            return -np.log(sys.float_info.min)

    loss = 0
    for i in range(probs.shape[0]):
        row = probs[i]
        index = target_index[i]
        loss -= np.log(row[index])

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    is_batch = len(predictions.shape) > 1

    softmax_out = softmax(predictions)
    loss = cross_entropy_loss(softmax_out, target_index)
    grad = softmax_out.copy()

    if is_batch:
        batch_size = softmax_out.shape[0]
        for i in range(batch_size):
            grad[i][target_index[i]] = softmax_out[i][target_index[i]] - 1
    else:
        grad[target_index] = softmax_out[target_index] - 1

    return loss, grad


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(np.power(W, 2))
    grad = W * 2 * reg_strength

    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    loss, grad = softmax_with_cross_entropy(predictions, target_index)

    dW = np.dot(X.T, grad)

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            loss = 0
            for indices in batches_indices:
                train = np.take(X, indices, axis=0)

                # TODO implement generating batches from indices
                # Compute loss and gradients
                # Apply gradient to weights using learning rate
                # Don't forget to add both cross-entropy loss
                # and regularization!

                loss_linear, dW_linear = linear_softmax(train, self.W, y)
                loss_l2, dW_l2 = l2_regularization(self.W, reg)

                loss = loss_linear + loss_l2
                dW = dW_linear + dW_l2

                self.W -= learning_rate * dW

            # end
            loss_history.append((epoch, loss))
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        predictions = softmax(np.dot(X, self.W))
        for i in range(predictions.shape[0]):
            max_key = 0
            max_val = -1

            for j in range(10):
                if predictions[i][j] > max_val:
                    max_val = predictions[i][j]
                    max_key = j

            y_pred[i] = max_key

        return y_pred

