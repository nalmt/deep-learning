# coding: utf8
# !/usr/bin/env python
import gzip, torch

BATCH_SIZE = 5
NB_EPOCHS = 30 # Nombre de fois que la base de données sera lue.
ETA = 0.0001 # taux d'apprentissage

if __name__ == '__main__':
    # On lit les données.
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('sujet/mnist.pkl.gz'))

    # On crée les lecteurs de données.
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    entry_layer = torch.nn.Linear(data_train.shape[1], 37)
    middle_layer_1 = torch.nn.Linear(37, 37)
    middle_layer_2 = torch.nn.Linear(37, 37)

    output_layer = torch.nn.Linear(37, label_train.shape[1])

    model = torch.nn.Sequential(entry_layer, torch.nn.ReLU(), middle_layer_1, torch.nn.ReLU(), middle_layer_2, torch.nn.ReLU(), output_layer)

    torch.nn.init.xavier_uniform_(entry_layer.weight)
    torch.nn.init.xavier_uniform_(output_layer.weight)
    torch.nn.init.xavier_uniform_(middle_layer_1.weight)
    torch.nn.init.xavier_uniform_(middle_layer_2.weight)

    optim = torch.optim.Adam(model.parameters(), lr=ETA)
    loss_func = torch.nn.MSELoss(reduction='sum')

    for n in range(NB_EPOCHS):
        # On lit toutes les données d'apprentissage.
        for (x, t) in train_loader:
            print(x, t)
            # Activation des neuronnes couche par couche.
            optim.zero_grad()
            y = model(x)
            print("y:", y)
            # Calcul de l'erreur delta.
            loss = loss_func(t, y)
            loss.backward()
            optim.step()
            # Mise à jour des poids w.

        # Test du modèle (on évalue la progression pendant l'apprentissage).
        acc = 0.
        # On lit toutes les donnéees de test.
        for (x, t) in test_loader:
            # on calcule la sortie du modèle
            y = model(x)
            # On regarde si la sortie est correcte.
            acc += torch.argmax(y, 1) == torch.argmax(t, 1)
        # On affiche le pourcentage de bonnes réponses.
        print(acc/data_test.shape[0])