class MSE_Loss():
    def __init__(self):
        self.loss = 0.0
        self.grad = 1.0
        self.input = 0.0
        self.name = "loss_MSE"
        self.y_true = ""

    def __call__(self, y_pred, y_true):
        self.input = y_pred
        self.y_true = y_true

        # Important to have pred - true
        # Otherwise gradient should be multiplied with -1
        self.loss = 0.0
        for pred, true in zip(self.input, self.y_true):
            self.loss += float((pred.data - true)**2)
        return self.loss

    def mse_derivative(self, x):
        return 2 * x

    def backward(self):
        input_grad = 0.0

        # Compute loss grad with respect to the inputs
        for pred, true in zip(self.input, self.y_true):
            input_grad += float(self.mse_derivative((pred.data - true)))

        # Set the inputs grad
        for pred in self.input:
            pred.grad += input_grad

    def zero_grad(self):
        self.input_grad = 0.0
