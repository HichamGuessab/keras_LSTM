import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Dictionnaire d'encodage pour les acides aminés
amino_acid_to_index = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}

# Dictionnaire structures secondaires
structure_to_index = {
    '_': 0,  # Pelote aléatoire
    'e': 1,  # Feuillet bêta
    'h': 2   # Hélice alpha
}

def load_data(file_path):
    sequences = []
    structures = []
    current_sequence = []
    current_structure = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "<>":  # debut ou la fin de la séquence
                if current_sequence and current_structure:
                    sequences.append(current_sequence)
                    structures.append(current_structure)
                current_sequence = []
                current_structure = []
            elif line:
                try:
                    amino_acid, structure = line.split()
                    current_sequence.append(amino_acid)
                    current_structure.append(structure)
                except ValueError:
                    continue  # on ignore les lignes incorrectes

    if current_sequence and current_structure:
        sequences.append(current_sequence)
        structures.append(current_structure)

    return sequences, structures

# Charger les données d'entraînement et de test
train_sequences, train_structures = load_data('Proteines/data/protein-secondary-structure.train')
test_sequences, test_structures = load_data('Proteines/data/protein-secondary-structure.test')

# Fonction pour convertir les séquences en indices
def convert_sequences_to_indices(sequences, dictionary):
    return [[dictionary[aa] for aa in seq] for seq in sequences]

# Conversion des données en indices
train_input_indices = convert_sequences_to_indices(train_sequences, amino_acid_to_index)
train_output_indices = convert_sequences_to_indices(train_structures, structure_to_index)
test_input_indices = convert_sequences_to_indices(test_sequences, amino_acid_to_index)
test_output_indices = convert_sequences_to_indices(test_structures, structure_to_index)

print("Petit exemple des familles de séquence encodée : \n", train_input_indices[:1], "\n", train_output_indices[:1])

# Paramètres
num_amino_acids = len(amino_acid_to_index)  # 20
num_structures = len(structure_to_index)    # 3
max_sequence_length = max(len(seq) for seq in train_input_indices)

# Padding des séquences d'entrainement et de test
train_input_padded = pad_sequences(train_input_indices, maxlen=max_sequence_length, padding='post')
train_output_padded = pad_sequences(train_output_indices, maxlen=max_sequence_length, padding='post')
test_input_padded = pad_sequences(test_input_indices, maxlen=max_sequence_length, padding='post')
test_output_padded = pad_sequences(test_output_indices, maxlen=max_sequence_length, padding='post')

# Conversion des sorties en one-hot encoding
train_output_categorical = to_categorical(train_output_padded, num_classes=num_structures)
test_output_categorical = to_categorical(test_output_padded, num_classes=num_structures)

# Construction du modèle
model = Sequential()
model.add(Embedding(input_dim=num_amino_acids, output_dim=64, input_length=max_sequence_length)) # Encodage des indices des acides aminés en vecteurs
model.add(LSTM(128, return_sequences=True)) # Traitement séquentiel pour capter les relations entre les acides aminés
model.add(TimeDistributed(Dense(num_structures, activation='softmax'))) # prédire la structure secondaire pour chaque position de la séquence

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Résumé du modèle
model.summary()

# Entraînement du modèle
model.fit(train_input_padded, train_output_categorical, epochs=10, batch_size=1)

# Evaluation du modèle sur les données de train
# On s'attend à ce que le modèle ait une meilleure performance sur les données de train que sur les données de test
train_loss, train_accuracy = model.evaluate(train_input_padded, train_output_categorical)
print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
# Train Loss: 0.3115890324115753
# Train Accuracy: 0.8540468215942383

# Evaluation du modèle sur les données de test
test_loss, test_accuracy = model.evaluate(test_input_padded, test_output_categorical)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# Test Loss: 0.41311725974082947
# Test Accuracy: 0.8017954230308533

# Prédiction avec le modèle
predictions = model.predict(train_input_padded)
print(predictions)