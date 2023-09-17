import math


class BaseLoss():
    def __init__(self):
        self.loss = 0.0
        self.grad = 1.0
        self.input = 0.0
        self.y_true = ""

    def backward(self):
        input_grad = []

        # Compute loss grad with respect to the inputs
        for pred, true in zip(self.input, self.y_true):
            input_grad.append(float(self.derivative(pred.data, true)))

        # Set the inputs grad
        for i in range(len(self.input)):
            self.input[i].grad += input_grad[i]

    def derivative(self, pred, true):
        pass


class MSE(BaseLoss):
    def __init__(self):
        self.name = "MSE"
        super().__init__()

    def __call__(self, y_pred, y_true):
        self.input = y_pred
        self.y_true = y_true

        # Important to have pred - true
        # Otherwise gradient should be multiplied with -1
        self.loss = 0.0
        for pred, true in zip(self.input, self.y_true):
            self.loss += float((pred.data - true)**2)
        return self.loss

    def derivative(self, pred, true):
        return 2 * (pred - true)


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        self.name = "CrossEntropyLoss"
        self.epsilon = 1e-15  # Small constant to avoid division by zero
        super().__init__()

    def __call__(self, y_pred, y_true):
        # Clip values between (epsilon, 1 - epsilon)
        for i in range(len(y_pred)):
            y_pred[i].data = max(min(1 - self.epsilon, y_pred[i].data), self.epsilon)

        self.input = y_pred
        self.y_true = y_true

        # Loss
        self.loss = 0.0
        for i in range(len(y_pred)):
            a = y_true[i] * math.log(float(y_pred[i].data))
            b = (1 - y_true[i]) * math.log(1 - float(y_pred[i].data))
            self.loss += (a + b)
        self.loss *= -1

        return self.loss

    def derivative(self, pred, true):
        return -(true / pred - (1 - true) / (1 - pred))
