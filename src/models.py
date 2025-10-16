import numpy as np

class ManualLinearRegression:
    def __init__(self, alpha=0.0, regularization='none', learning_rate=0.01, max_iters=1000):
        self.alpha = alpha
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.loss_history = []
    def normal_equation(self, X, y):
        try:
            if self.regularization == 'l2' and self.alpha > 0:
                identity = np.eye(X.shape[1])
                identity[0, 0] = 0
                self.weights = np.linalg.pinv(X.T @ X + self.alpha * identity) @ X.T @ y
            else:
                self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
            
    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for i in range(self.max_iters):
            y_pred = X @ self.weights  
            if self.regularization == 'l2':
                gradient = (2/n_samples) * (X.T @ (y_pred - y) + self.alpha * self.weights)
                gradient[0] = (2/n_samples) * (X[:, 0] @ (y_pred - y))
            elif self.regularization == 'l1':
                gradient = (2/n_samples) * (X.T @ (y_pred - y))
                gradient[1:] += (self.alpha/n_samples) * np.sign(self.weights[1:])
            else:
                gradient = (2/n_samples) * (X.T @ (y_pred - y))  
            self.weights -= self.learning_rate * gradient
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                break
                
    def compute_loss(self, X, y):
        y_pred = X @ self.weights
        mse = np.mean((y_pred - y) ** 2)
        if self.regularization == 'l2':
            reg_term = self.alpha * np.sum(self.weights[1:] ** 2)
        elif self.regularization == 'l1':
            reg_term = self.alpha * np.sum(np.abs(self.weights[1:]))
        else:
            reg_term = 0
        return mse + reg_term
        
    def fit(self, X, y, method='normal'):
        if method == 'normal':
            self.normal_equation(X, y)
        else:
            self.gradient_descent(X, y) 
    def predict(self, X):
        return X @ self.weights
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    def get_coefficients(self):
        """Return the model coefficients (including bias term)"""
        return self.weights if self.weights is not None else None
    def get_loss_history(self):
        """Return the loss history from gradient descent"""
        return self.loss_history