# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron multi-couches en pytorch (en utilisant juste les tenseurs)
# Écrit par Nabil Lamrabet
# ------------------------------------------------------------------------
from neuron import *

BATCH_SIZE = 5 # nombre de données lues à chaque fois.
NB_EPOCHS = 10 # nombre de fois que la base de données sera lue.

if __name__ == '__main__':
    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('sujet/mnist.pkl.gz'))

    nb_data_train = data_train.shape[0]
    nb_data_test = data_test.shape[0]
    indices = numpy.arange(nb_data_train, step=BATCH_SIZE)

    # On stockera par la suite dans cette variable l'image en entrée.
    # Une image est un vecteur de dimension [784], pour des raisons pratiques
    # on souhaite avoir l'image sous la forme d'une matrice de dimension [1, 784].
    data = torch.empty((1, data_train[0].shape[0]), dtype=torch.float)

    entry_layer = EntryLayer(3, sigmoid_activation, data_train[0].shape[0])
    output_layer = OutputLayer(10, linear_activation, entry_layer)

    entry_layer.next_layer = output_layer

    for n in range(NB_EPOCHS):
        numpy.random.shuffle(indices)

        for image in indices:
            data[0] = data_train[image:image+1]

            # Activation des neuronnes couche par couche.
            entry_layer.activate(data)
            output_layer.activate()

            # Calcul de l'erreur delta et rétroprogragation.
            output_layer.calculate_delta_error(label_train[image:image+1])
            entry_layer.calculate_delta_error()

            # Mise à jour des poids w.
            output_layer.update_w()
            entry_layer.update_w(data)

		# Test du modèle (on évalue la progression pendant l'apprentissage).
        acc = 0.
		# On lit toutes les donnéees de test.
        for i in range(nb_data_test):
            data[0] = data_test[i:i+1]
            entry_layer.activate(data)
            output_layer.activate()
            # On regarde le vrai label.
            t = label_test[i:i+1]
            # On regarde si la sortie est correcte.
            acc += torch.argmax(output_layer.y, 1) == torch.argmax(t, 1)
        # On affiche le pourcentage de bonnes réponses.
        print(acc/nb_data_test)