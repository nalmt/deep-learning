# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron multi-couches en pytorch (en utilisant juste les tenseurs)
# Écrit par Nabil Lamrabet
# ------------------------------------------------------------------------
from neuron import *

NB_EPOCHS = 10 # nombre de fois que la base de données sera lue.

if __name__ == '__main__':
    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('sujet/mnist.pkl.gz'))

    nb_data_train = data_train.shape[0]
    nb_data_test = data_test.shape[0]
    indices = numpy.arange(nb_data_train, step=BATCH_SIZE)

    single_layer = SingleLayer(10, linear_activation, data_train[0].shape[0])

    for n in range(NB_EPOCHS):
        # on mélange les (indices des) données
        numpy.random.shuffle(indices)
        # on lit toutes les données d'apprentissage
        for i in indices:
            # on récupère les entrées
            x = data_train[i:i+BATCH_SIZE]
            # Activation des neuronnes couche par couche.
            single_layer.activate(x)
            # on regarde les vrais labels
            t = label_train[i:i+BATCH_SIZE]
            # Calcul de l'erreur delta.
            single_layer.calculate_delta_error(t)
            # Mise à jour des poids w.
            single_layer.update_wb(x)

        # Test du modèle (on évalue la progression pendant l'apprentissage).
        acc = 0.
        # On lit toutes les donnéees de test.
        for i in range(nb_data_test):
            # on récupère l'entrée
            x = data_test[i:i+1]
            # on calcule la sortie du modèle
            single_layer.activate(x)
            # On regarde le vrai label.
            t = label_test[i:i+1]
            # On regarde si la sortie est correcte.
            acc += torch.argmax(single_layer.y, 1) == torch.argmax(t, 1)
        # On affiche le pourcentage de bonnes réponses.
        print(acc/nb_data_test)