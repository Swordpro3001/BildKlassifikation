"""
Bildklassifikation mit TensorFlow
Aufgabe: Klassifikation von handgeschriebenen Mathematiksymbolen
Author: Data Science Project
Date: Oktober 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import load_img, img_to_array
import random
from pathlib import Path
import kagglehub

# Setze Random Seeds für Reproduzierbarkeit
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Download dataset
print("Lade Datensatz herunter...")
dataset_path = kagglehub.dataset_download("xainano/handwrittenmathsymbols")
print(f"Path to dataset files: {dataset_path}")

# Konfiguration
SYMBOL1 = '+'  # Erstes Symbol (koordiniere mit Kollegen!)
SYMBOL2 = '-'  # Zweites Symbol (koordiniere mit Kollegen!)
IMG_SIZE = (45, 45)  # Original size der Bilder
SAMPLES_PER_SYMBOL = 30
TRAIN_SAMPLES = 20
TEST_SAMPLES = 10
EPOCHS_APPROACH1 = 1000
EPOCHS_APPROACH2 = 1000


class DataLoader:
    """Klasse zum Laden und Vorbereiten der Bilddaten"""
    
    def __init__(self, dataset_path, symbol1, symbol2, samples_per_symbol=30):
        self.dataset_path = Path(dataset_path)
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.samples_per_symbol = samples_per_symbol
        
    def load_images(self):
        """Lädt die Bilder für beide Symbole"""
        print(f"\nLade Bilder für Symbole: {self.symbol1} und {self.symbol2}")
        
        # Finde alle Bilder für beide Symbole
        all_files = list(self.dataset_path.rglob("*.png")) + \
                   list(self.dataset_path.rglob("*.jpg"))
        
        symbol1_images = []
        symbol2_images = []
        
        # Lade Bilder auf Pixelbasis
        for img_file in all_files:
            filename = img_file.name.lower()
            
            # Prüfe, welches Symbol im Dateinamen vorkommt
            if self.symbol1.lower() in filename or \
               self._get_symbol_name(self.symbol1) in filename:
                if len(symbol1_images) < self.samples_per_symbol:
                    img_array = self._load_and_preprocess(img_file)
                    if img_array is not None:
                        symbol1_images.append(img_array)
                        
            elif self.symbol2.lower() in filename or \
                 self._get_symbol_name(self.symbol2) in filename:
                if len(symbol2_images) < self.samples_per_symbol:
                    img_array = self._load_and_preprocess(img_file)
                    if img_array is not None:
                        symbol2_images.append(img_array)
            
            # Früher Abbruch wenn genug Bilder
            if len(symbol1_images) >= self.samples_per_symbol and \
               len(symbol2_images) >= self.samples_per_symbol:
                break
        
        print(f"Geladene Bilder - {self.symbol1}: {len(symbol1_images)}, "
              f"{self.symbol2}: {len(symbol2_images)}")
        
        return np.array(symbol1_images), np.array(symbol2_images)
    
    def _load_and_preprocess(self, img_path):
        """Lädt ein Bild und bereitet es vor"""
        try:
            # Lade Bild
            img = load_img(img_path, target_size=IMG_SIZE, color_mode='grayscale')
            # Konvertiere zu Array
            img_array = img_to_array(img)
            # Normalisiere auf [0, 1]
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            print(f"Fehler beim Laden von {img_path}: {e}")
            return None
    
    def _get_symbol_name(self, symbol):
        """Hilfsfunktion für Symbolnamen"""
        symbol_names = {
            '+': 'plus',
            '-': 'minus',
            'x': 'times',
            '/': 'div',
            '=': 'eq',
            '(': 'lparen',
            ')': 'rparen',
        }
        return symbol_names.get(symbol, symbol)


def prepare_data(symbol1_imgs, symbol2_imgs, train_size=20):
    """Bereitet Trainings- und Testdaten vor"""
    # Labels erstellen (0 für Symbol1, 1 für Symbol2)
    y1 = np.zeros(len(symbol1_imgs))
    y2 = np.ones(len(symbol2_imgs))
    
    # Kombiniere Daten
    X = np.concatenate([symbol1_imgs, symbol2_imgs])
    y = np.concatenate([y1, y2])
    
    # Shuffle für zufällige Aufteilung
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    split_idx = train_size * 2  # 2 Klassen
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nDaten vorbereitet:")
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def create_model(num_hidden_layers=2, units_per_layer=64):
    """
    Erstellt ein CNN-Modell mit konfigurierbarer Anzahl verdeckter Schichten
    
    Args:
        num_hidden_layers: Anzahl der Dense Hidden Layers nach Conv-Layers
        units_per_layer: Anzahl der Neuronen pro Hidden Layer
    """
    model = keras.Sequential([
        # Input Layer
        layers.Input(shape=(*IMG_SIZE, 1)),
        
        # Convolutional Layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten
        layers.Flatten(),
    ])
    
    # Dynamisch Hidden Layers hinzufügen
    for i in range(num_hidden_layers):
        model.add(layers.Dense(units_per_layer, activation='relu'))
        model.add(layers.Dropout(0.3))  # Dropout zur Regularisierung
    
    # Output Layer (Binary Classification)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Kompiliere Modell
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def approach1_sequential_training(X_train, y_train, X_test, y_test, epochs=1000):
    """
    Ansatz 1: Sequentielles Training
    - Erst alle Bilder von Symbol 1 trainieren
    - Dann alle Bilder von Symbol 2 trainieren
    """
    print("\n" + "="*70)
    print("ANSATZ 1: Sequentielles Training")
    print("="*70)
    
    model = create_model(num_hidden_layers=2)
    
    # Sortiere Trainingsdaten nach Labels
    sort_idx = np.argsort(y_train)
    X_train_sorted = X_train[sort_idx]
    y_train_sorted = y_train[sort_idx]
    
    # Finde Split-Punkt zwischen den Klassen
    split_point = np.sum(y_train_sorted == 0)
    
    # Trainiere erst Symbol 1 (Label 0)
    print(f"\nTrainiere {epochs} Epochen mit Symbol 1...")
    X_symbol1 = X_train_sorted[:split_point]
    y_symbol1 = y_train_sorted[:split_point]
    
    history1 = model.fit(
        X_symbol1, y_symbol1,
        epochs=epochs,
        batch_size=1,
        verbose=0
    )
    
    # Dann trainiere Symbol 2 (Label 1)
    print(f"Trainiere {epochs} Epochen mit Symbol 2...")
    X_symbol2 = X_train_sorted[split_point:]
    y_symbol2 = y_train_sorted[split_point:]
    
    history2 = model.fit(
        X_symbol2, y_symbol2,
        epochs=epochs,
        batch_size=1,
        verbose=0
    )
    
    # Evaluierung
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy (Ansatz 1): {test_acc:.4f}")
    
    return model, history1, history2, test_acc


def approach2_mixed_training(X_train, y_train, X_test, y_test, epochs=1000):
    """
    Ansatz 2: Gemischtes Training
    - In jeder Epoche ein zufälliges Symbol aus den Trainingsdaten
    """
    print("\n" + "="*70)
    print("ANSATZ 2: Gemischtes Training (Shuffled)")
    print("="*70)
    
    model = create_model(num_hidden_layers=2)
    
    # Durchmische die Trainingsdaten
    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]
    
    print(f"\nTrainiere {epochs} Epochen mit gemischten Symbolen...")
    history = model.fit(
        X_train_shuffled, y_train_shuffled,
        epochs=epochs,
        batch_size=1,
        verbose=0,
        validation_data=(X_test, y_test)
    )
    
    # Evaluierung
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy (Ansatz 2): {test_acc:.4f}")
    
    return model, history, test_acc


def compare_architectures(X_train, y_train, X_test, y_test, 
                          hidden_layers_configs=[1, 2, 4]):
    """
    Vergleicht verschiedene Netzwerkarchitekturen mit unterschiedlicher
    Anzahl verdeckter Schichten
    """
    print("\n" + "="*70)
    print("ARCHITEKTUR-VERGLEICH: Verschiedene Anzahlen verdeckter Schichten")
    print("="*70)
    
    results = {}
    models = {}
    
    for num_layers in hidden_layers_configs:
        print(f"\n--- Modell mit {num_layers} verdeckten Schichten ---")
        
        model = create_model(num_hidden_layers=num_layers, units_per_layer=64)
        
        # Training mit gemischten Daten
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Weniger Epochen für schnelleren Vergleich
            batch_size=4,
            verbose=0,
            validation_data=(X_test, y_test)
        )
        
        # Evaluierung
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        results[num_layers] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'history': history
        }
        models[num_layers] = model
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
    
    return models, results


def visualize_predictions(model, X_test, y_test, num_samples=10):
    """Visualisiert Vorhersagen des Modells"""
    predictions = model.predict(X_test)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(X_test))):
        axes[i].imshow(X_test[i].squeeze(), cmap='gray')
        
        true_label = "Symbol 2" if y_test[i] == 1 else "Symbol 1"
        pred_label = "Symbol 2" if predictions[i] > 0.5 else "Symbol 1"
        confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.2f})", 
                         color=color, fontsize=9)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("\nVisualisierung gespeichert als 'predictions.png'")
    plt.close()


def plot_comparison(results_dict):
    """Erstellt Vergleichsplots für verschiedene Konfigurationen"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = list(results_dict.keys())
    accuracies = [results_dict[l]['accuracy'] for l in layers]
    losses = [results_dict[l]['loss'] for l in layers]
    
    # Accuracy Vergleich
    ax1.bar(range(len(layers)), accuracies, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Anzahl verdeckter Schichten', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Genauigkeit nach Anzahl Schichten', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Loss Vergleich
    ax2.bar(range(len(layers)), losses, color='coral', alpha=0.7)
    ax2.set_xlabel('Anzahl verdeckter Schichten', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Verlust nach Anzahl Schichten', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
    print("Architektur-Vergleich gespeichert als 'architecture_comparison.png'")
    plt.close()


def main():
    """Hauptfunktion zur Ausführung aller Experimente"""
    print("="*70)
    print("BILDKLASSIFIKATION MIT TENSORFLOW")
    print("Handgeschriebene Mathematiksymbole")
    print("="*70)
    
    # 1. Daten laden
    loader = DataLoader(dataset_path, SYMBOL1, SYMBOL2, SAMPLES_PER_SYMBOL)
    symbol1_imgs, symbol2_imgs = loader.load_images()
    
    if len(symbol1_imgs) < SAMPLES_PER_SYMBOL or len(symbol2_imgs) < SAMPLES_PER_SYMBOL:
        print(f"\nWARNUNG: Nicht genug Bilder gefunden!")
        print(f"Benötigt: {SAMPLES_PER_SYMBOL} pro Symbol")
        print(f"Gefunden: {len(symbol1_imgs)} ({SYMBOL1}), {len(symbol2_imgs)} ({SYMBOL2})")
        print("\nÄndere SYMBOL1 und SYMBOL2 in der Konfiguration!")
        return
    
    # 2. Daten vorbereiten
    X_train, X_test, y_train, y_test = prepare_data(
        symbol1_imgs, symbol2_imgs, 
        train_size=TRAIN_SAMPLES
    )
    
    # 3. EXPERIMENT 1: Ansatz 1 vs Ansatz 2
    print("\n" + "="*70)
    print("EXPERIMENT 1: Vergleich der Trainingsansätze")
    print("="*70)
    
    model1, hist1_1, hist1_2, acc1 = approach1_sequential_training(
        X_train, y_train, X_test, y_test, epochs=100  # Reduziert für Demo
    )
    
    model2, hist2, acc2 = approach2_mixed_training(
        X_train, y_train, X_test, y_test, epochs=100  # Reduziert für Demo
    )
    
    print(f"\n--- ERGEBNISSE EXPERIMENT 1 ---")
    print(f"Ansatz 1 (Sequentiell):  {acc1:.4f}")
    print(f"Ansatz 2 (Gemischt):     {acc2:.4f}")
    print(f"Unterschied:             {abs(acc2 - acc1):.4f}")
    
    # 4. EXPERIMENT 2: Verschiedene Architekturen
    print("\n" + "="*70)
    print("EXPERIMENT 2: Vergleich der Architekturen")
    print("="*70)
    
    models_arch, results_arch = compare_architectures(
        X_train, y_train, X_test, y_test,
        hidden_layers_configs=[1, 3, 5]
    )
    
    # 5. Visualisierungen
    print("\n" + "="*70)
    print("VISUALISIERUNGEN")
    print("="*70)
    
    # Vorhersagen visualisieren (beste Modell verwenden)
    best_layers = max(results_arch.keys(), 
                     key=lambda k: results_arch[k]['accuracy'])
    best_model = models_arch[best_layers]
    visualize_predictions(best_model, X_test, y_test)
    
    # Architektur-Vergleich plotten
    plot_comparison(results_arch)
    
    # 6. Zusammenfassung
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)
    print(f"\nSymbole: {SYMBOL1} vs {SYMBOL2}")
    print(f"Trainingssamples: {TRAIN_SAMPLES} pro Klasse")
    print(f"Testsamples: {TEST_SAMPLES} pro Klasse")
    print(f"\nBeste Architektur: {best_layers} verdeckte Schichten")
    print(f"Beste Accuracy: {results_arch[best_layers]['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*70)


if __name__ == "__main__":
    main()