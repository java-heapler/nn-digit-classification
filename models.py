import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dot_product = self.run(x)
        score = nn.as_scalar(dot_product)
        return 1 if score >= 0 else -1        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.01
        prev_weights = self.w.data.copy()
        while True:
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                if prediction != nn.as_scalar(y):
                    # Update the weights using the perceptron update rule
                    self.w.data += learning_rate * nn.as_scalar(y) * x.data

            # Check for convergence
            if np.linalg.norm(self.w.data - prev_weights) < 1e-5:
                break
            prev_weights = self.w.data.copy()
            
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(1, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        self.w2 = nn.Parameter(hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        hidden_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        return nn.AddBias(nn.Linear(hidden_layer, self.w2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.01
        num_epochs = 1000

        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

                self.w1.update(grad_wrt_w1, -learning_rate)
                self.b1.update(grad_wrt_b1, -learning_rate)
                self.w2.update(grad_wrt_w2, -learning_rate)
                self.b2.update(grad_wrt_b2, -learning_rate)

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}")        

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(784, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        self.w2 = nn.Parameter(hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        hidden_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        return nn.AddBias(nn.Linear(hidden_layer, self.w2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_scores = self.run(x)
        return nn.SoftmaxLoss(predicted_scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.01
        num_epochs = 10

        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in dataset.iterate_once(100):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

                self.w1.update(grad_wrt_w1, -learning_rate)
                self.b1.update(grad_wrt_b1, -learning_rate)
                self.w2.update(grad_wrt_w2, -learning_rate)
                self.b2.update(grad_wrt_b2, -learning_rate)

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}")
            
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # RNN parameters
        self.w_xh = nn.Parameter(self.num_chars, hidden_size)
        self.w_hh = nn.Parameter(hidden_size, hidden_size)
        self.b_h = nn.Parameter(1, hidden_size)

        # Linear layer for language scores
        self.w_hy = nn.Parameter(hidden_size, len(self.languages))
        self.b_y = nn.Parameter(1, len(self.languages))
        
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        batch_size = xs[0].data.shape[0]
        h = nn.Tensor(batch_size, self.hidden_size).zeroes()

        for x in xs:
            # Apply one step of the RNN
            h = nn.ReLU(nn.Add(nn.AddBias(nn.Linear(x, self.w_xh), self.b_h), nn.Linear(h, self.w_hh)))

        return nn.AddBias(nn.Linear(h, self.w_hy), self.b_y)
    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_scores = self.run(xs)
        return nn.SoftmaxLoss(predicted_scores, y)
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.01
        num_epochs = 10

        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in dataset.iterate_once(100):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                grad_wrt_w_xh, grad_wrt_w_hh, grad_wrt_b_h, grad_wrt_w_hy, grad_wrt_b_y = nn.gradients(
                    loss, [self.w_xh, self.w_hh, self.b_h, self.w_hy, self.b_y])

                self.w_xh.update(grad_wrt_w_xh, -learning_rate)
                self.w_hh.update(grad_wrt_w_hh, -learning_rate)
                self.b_h.update(grad_wrt_b_h, -learning_rate)
                self.w_hy.update(grad_wrt_w_hy, -learning_rate)
                self.b_y.update(grad_wrt_b_y, -learning_rate)

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}")