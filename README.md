Nabil Lamrabet
Emna Barred

26/11/20
03/12/20

# TP Introduction à l'apprentissage profond

# Partie 1 : Perceptron

- Indiquer la taille de chaque tenseur dans le premier fichier. Expliquer.

On sait que les images 28x28 sont stockés sous forme de vecteur de dimension [784] et les labels sous forme de vecteur de dimension [10].

data_train : nombre images * image[784]
label_train : nombre images * label[10]
data_test : nombre images * image[28][28]
label_test : nombre images * label[10]

data_train = [[image_1[784], image_2[784], ..., image_63_000[784]]
label_train = [[label_1[10], label_2[10], ..., label_63_000[10]]]

data_test = [[image_1[784], image2_[784], ..., image_63_000[784]]
label_test = [[label_1[10], label_2[10], ..., label_63_000[10]]

l'attribut `shape` des vecteurs retourne les dimensions.

Par exemple data_train.shape = [63000, 784] et label_train.shape = [63000, 10]

les dimensions de w sont [28, 10]
les dimensions de b sont [1, 10]

- Quantifier l’impact des différents hyperparamètres (en particulier eta et les poids initiaux) sur les performances.

TODO

# Partie 2 : Shallow network

Implémentation en cours