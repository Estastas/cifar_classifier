import numpy as np
import torch


class Trainer:

    @staticmethod
    def train(net, X_train, y_train, X_test, y_test):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)

        batch_size = 100

        test_accuracy_history = []
        test_loss_history = []

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        for epoch in range(30):
            order = np.random.permutation(len(X_train))
            for start_index in range(0, len(X_train), batch_size):
                optimizer.zero_grad()
                net.train()

                batch_indexes = order[start_index:start_index + batch_size]

                X_batch = X_train[batch_indexes].to(device)
                y_batch = y_train[batch_indexes].to(device)

                preds = net.forward(X_batch)

                loss_value = loss(preds, y_batch)
                loss_value.backward()

                optimizer.step()

            net.eval()
            test_preds = net.forward(X_test)
            test_loss_history.append(loss(test_preds, y_test).data.cpu())

            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
            test_accuracy_history.append(accuracy)

        return net, test_accuracy_history, test_loss_history
