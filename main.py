import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Exemple de données
# Les séquences d'acides aminés sont représentées par des indices de 0 à 19
# Les structures secondaires sont représentées par des indices de 0 à 2

# Données d'entrée (séquences d'acides aminés)
input_sequences = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14]
]

# Données de sortie (structures secondaires)
output_sequences = [
    [0, 1, 2, 0, 1],
    [2, 0, 1, 2, 0],
    [1, 2, 0, 1, 2]
]

# Paramètres
num_amino_acids = 20
num_structures = 3
max_sequence_length = max(len(seq) for seq in input_sequences)

# Préparation des données
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

# Conversion des sorties en one-hot encoding
output_sequences = to_categorical(output_sequences, num_classes=num_structures)

# Construction du modèle
model = Sequential()
model.add(Embedding(input_dim=num_amino_acids, output_dim=64, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(num_structures, activation='softmax')))

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Résumé du modèle
model.summary()

# Entraînement du modèle
model.fit(input_sequences, output_sequences, epochs=10, batch_size=1)

# Prédiction avec le modèle
predictions = model.predict(input_sequences)
print(predictions)