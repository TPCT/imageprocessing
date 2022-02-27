import torch
from math import e


class GenericFunctions:
    @staticmethod
    def exponentialSum(vector):
        accumulator = 0
        for element in vector:
            accumulator += e ** element
        return accumulator

    @staticmethod
    def softmax(input_vector):
        for i in range(input_vector.shape[0]):
            output = torch.zeros(input_vector.shape)

            if len(input_vector.shape) == 1:
                row_exp_sum = GenericFunctions.exponentialSum(input_vector)
            else:
                row_exp_sum = GenericFunctions.exponentialSum(input_vector[i])

            for j in range(input_vector.shape[-1]):
                if len(input_vector.shape) == 1:
                    output[j] = ((e ** input_vector[j]) / row_exp_sum)
                else:
                    output[i][j] = ((e ** input_vector[i][j]) / row_exp_sum)

            return output

    @staticmethod
    def accuracy(outputs, targets):
        _, predictions = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(predictions == targets).item() / len(predictions))

    @staticmethod
    def fit(iterations, learning_rate, model, train_loader, validation_loader, optimizer_function=torch.optim.SGD, lock=None):
        optimizer = optimizer_function(model.parameters(), learning_rate)
        history = []

        for iteration in range(iterations):
            for batch in train_loader:
                loss = model.trainingStep(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = GenericFunctions.evaluate(model, validation_loader)
            if lock:
                with lock:
                    GenericFunctions.iterationEnd(iteration, result)
            else:
                GenericFunctions.iterationEnd(iteration, result)
            history.append(result)

        return history

    @staticmethod
    def evaluate(model, val_loader):
        outputs = [model.validationStep(batch) for batch in val_loader]
        return GenericFunctions.validationIterationEnd(outputs)

    @staticmethod
    def validationIterationEnd(outputs):
        batch_losses = [x['loss_value'] for x in outputs]
        iteration_loss = torch.stack(batch_losses).mean()
        batch_accuracy = [x['accuracy_value'] for x in outputs]
        iteration_accuracy = torch.stack(batch_accuracy).mean()
        return {
            'loss_value': iteration_loss,
            'accuracy_value': iteration_accuracy
        }

    @staticmethod
    def iterationEnd(iteration, result):
        print("iteration [{}], average losses: {:.4f}, average accuracy: {:.4f}".format(iteration, result['loss_value'],
                                                                                        result['accuracy_value']))

    @staticmethod
    def predict(image, model):
        image = image.unsqueeze(0)
        output_prediction = model(image)
        _, prediction = torch.max(output_prediction, dim=1)
        return prediction[0].item()
