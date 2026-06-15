class Neuron:
    def __init__(self):
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0

    def predict(self, x1, x2):
        return self.w1 * x1 + self.w2 * x2 + self.b

    def train(self, dataset, learning_rate, max_itr):

        for itr in range(max_itr):

            total_loss = 0

            for x1, x2, y in dataset:

                prediction = self.predict(x1, x2)

                error = prediction - y

                loss = error * error
                total_loss += loss

                dw1 = 2 * error * x1
                dw2 = 2 * error * x2
                db = 2 * error

                self.w1 -= learning_rate * dw1
                self.w2 -= learning_rate * dw2
                self.b -= learning_rate * db

            if itr % 100 == 0:
                avg_loss = total_loss / len(dataset)
                print("itr:", itr, "Loss:", avg_loss)

    def show_parameters(self):
        print("w1 =", self.w1)
        print("w2 =", self.w2)
        print("b  =", self.b)


dataset = [
    [750, 1, 180],
    [800, 1, 190],
    [900, 2, 220],
    [1000, 2, 250],
    [1100, 2, 270],
    [1200, 3, 310],
    [1300, 3, 330],
    [1400, 3, 360],
    [1500, 4, 400],
    [1600, 4, 430],
    [1700, 4, 460],
    [1800, 5, 500],
    [1900, 5, 530],
    [2000, 5, 560],
    [2200, 6, 620]
]

neuron = Neuron()

print("Before Training")
neuron.show_parameters()

neuron.train(
    dataset,
    learning_rate=0.00000001,
    max_itr=5000
)

print("\nAfter Training")
neuron.show_parameters()

size = 1450
bedrooms = 3

predicted_price = neuron.predict(size, bedrooms)

print("\nPrediction")
print("Size:", size)
print("Bedrooms:", bedrooms)
print("Predicted Price:", round(predicted_price, 2))
