import torch

class LinearModel:
    def __init__(self):
        self.w = None 

    def score(self, X):
        if self.w is None: 
            # Initialize weights randomly with the same size as the number of features in X
            self.w = torch.rand((X.size()[1]))
        
        return torch.matmul(X, self.w)

    def predict(self, X):
        scores = self.score(X)
        y_hat = (scores > 0)
        return 1.0 * y_hat

class Perceptron(LinearModel):
    def loss(self, X, y):
        # Modify labels to be in {-1, 1}
        y_ = 2 * self.predict(X) - 1
        
        misclassified = 1.0*(y_ * y <0)
        return misclassified.mean()

    def grad(self, X, y):
       # Computes the vector to add to the weights to minimize the loss
       s = X@self.w
       return (s*y < 0)*X*y

class PerceptronOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y):

        # Randomly select a data point
        n = X.size()[0]
        i = torch.randint(n, size = (1,))

        # Subset 
        x_i = X[[i],:]
        y_i = y[i]

        # Record loss before change
        current_loss = self.model.loss(X, y)

        # Update weights
        self.model.w += torch.reshape(self.model.grad(x_i, y_i),(self.model.w.size()[0],))
        
        new_loss = self.model.loss(X, y)


        return i, abs(current_loss - new_loss)
