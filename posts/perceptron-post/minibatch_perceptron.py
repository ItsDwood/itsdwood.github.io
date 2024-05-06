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

class MiniBatchPerceptron(LinearModel):
    def loss(self, X, y):
        # Modify labels to be in {-1, 1}
        y_ = 2 * self.predict(X) - 1
        
        misclassified = 1.0*(y_ * y <0)
        return misclassified.mean()

    def grad(self, X, y, k, alpha):
       # Computes the vector to add to the weights to minimize the loss
        return (alpha/k)*(X*torch.reshape((X@self.w*y < 0)*y, (k,1))).sum(axis = 0)


class MiniBatchPerceptronOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y, k, alpha):

        # Randomly select a data point
        n = X.size()[0]
        ix = torch.randperm(n)[:k]

        # Subset 
        x_i = X[ix,:]
        y_i = y[ix]

        # Record loss before change
        current_loss = self.model.loss(X, y)

        # Update weights
        self.model.w += torch.reshape(self.model.grad(x_i, y_i, k, alpha),(self.model.w.size()[0],))
        
        new_loss = self.model.loss(X, y)


        return ix, abs(current_loss - new_loss)
