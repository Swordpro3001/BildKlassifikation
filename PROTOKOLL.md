# Laborprotokoll: Neuronale Netzwerke - TensorFlow

**Projekt:** Bildklassifikation mit Convolutional Neural Networks  
**Datum:** 16. Oktober 2025  
**Author:** Franz Puerto
**Klasse:** 5BHIT

---

## 1. Einführung

### 1.1 Zielsetzung

Das Ziel dieser Laborübung ist es, **Convolutional Neural Networks (CNNs)** anzupassen, um eine Klassifikation von handgeschriebenen Mathematiksymbolen durchzuführen. Dabei werden verschiedene Trainingsansätze und Netzwerkarchitekturen verglichen und evaluiert.

### 1.2 Theoretischer Hintergrund

#### Neuronale Netzwerke
Ein **neuronales Netzwerk** ist ein maschinelles Lernmodell, das von der Struktur biologischer Nervensysteme inspiriert ist. Es besteht aus miteinander verbundenen Knoten (Neuronen), die in Schichten organisiert sind:
- **Input Layer**: Empfängt die Eingabedaten
- **Hidden Layers**: Verarbeiten und extrahieren Features
- **Output Layer**: Erzeugt die finale Vorhersage

#### Tensoren
Ein **Tensor** ist eine mehrdimensionale Datenstruktur, die in Deep Learning zur Repräsentation von Daten verwendet wird:
- 0D-Tensor: Skalar (einzelne Zahl)
- 1D-Tensor: Vektor (Array)
- 2D-Tensor: Matrix (Tabelle)
- 3D-Tensor: z.B. Graustufenbild (Höhe × Breite × Kanäle)
- 4D-Tensor: Batch von Bildern (Batch × Höhe × Breite × Kanäle)

#### Layers (Schichten)
Verschiedene Arten von Schichten haben spezifische Funktionen:
- **Convolutional Layer**: Extrahiert räumliche Features aus Bildern durch Filter
- **Pooling Layer**: Reduziert die Dimensionalität und behält wichtige Features
- **Dense (Fully Connected) Layer**: Alle Neuronen mit allen Neuronen der vorherigen Schicht verbunden
- **Dropout Layer**: Regularisierungstechnik zur Vermeidung von Overfitting

#### Convolutional Neural Networks
CNNs sind speziell für die Verarbeitung von Bilddaten entwickelt und nutzen:
- **Konvolutionsfilter**: Erkennen lokale Muster (Kanten, Texturen)
- **Parameter-Sharing**: Reduziert die Anzahl zu lernender Parameter
- **Translationsinvarianz**: Erkennt Muster unabhängig von ihrer Position

---

## 2. Methodik

### 2.1 Datensatz

**Quelle:** [Handwritten Math Symbols Dataset (Kaggle)](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)

**Ausgewählte Symbole:** `+` (Plus) und `-` (Minus)

**Datenaufteilung:**
- 30 Bilder pro Symbol
- Trainingsdaten: 20 Bilder pro Symbol (40 gesamt)
- Testdaten: 10 Bilder pro Symbol (20 gesamt)

### 2.2 Datenvorverarbeitung

```python
# Bildgröße: 45x45 Pixel (Graustufenbild)
IMG_SIZE = (45, 45)

# Normalisierung auf [0, 1]
img_array = img_array / 255.0
```

Die Bilder werden:
1. In Graustufen konvertiert
2. Auf einheitliche Größe skaliert
3. Pixelwerte normalisiert (0-1 Range)
4. Als Numpy-Arrays gespeichert

### 2.3 Modellarchitektur

#### Basis CNN-Architektur

```python
model = keras.Sequential([
    # Convolutional Blocks
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten
    layers.Flatten(),
    
    # Hidden Layers (variabel)
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    # Output Layer
    layers.Dense(1, activation='sigmoid')
])
```

**Komponenten:**
- **Conv2D**: 2D-Konvolutionsschichten mit ReLU-Aktivierung
- **MaxPooling2D**: Reduziert räumliche Dimensionen um Faktor 2
- **Flatten**: Konvertiert 2D-Feature-Maps in 1D-Vektor
- **Dense**: Vollständig verbundene Schichten
- **Dropout**: Schaltet zufällig 30% der Neuronen ab (Regularisierung)
- **Sigmoid**: Outputaktivierung für binäre Klassifikation

**Verlustfunktion:** Binary Crossentropy  
**Optimizer:** Adam  
**Metrik:** Accuracy

---

## 3. Experimente

### 3.1 Experiment 1: Vergleich der Trainingsansätze

#### Ansatz 1: Sequentielles Training

**Methode:** 
- Trainiere zuerst 1000 Epochen nur mit Symbol 1 (+)
- Dann trainiere 1000 Epochen nur mit Symbol 2 (-)

**Hypothese:** Das Modell könnte "vergessen", was es über Symbol 1 gelernt hat, wenn es danach intensiv mit Symbol 2 trainiert wird (**Catastrophic Forgetting**).

```python
# Pseudo-Code
for epoch in range(1000):
    train_on_symbol1()

for epoch in range(1000):
    train_on_symbol2()
```

#### Ansatz 2: Gemischtes Training

**Methode:**
- In jeder Epoche werden die Trainingsdaten durchmischt
- Modell lernt beide Symbole gleichzeitig
- Verhindert Catastrophic Forgetting

```python
# Pseudo-Code
for epoch in range(1000):
    shuffle_data()
    train_on_mixed_symbols()
```

**Erwartetes Ergebnis:** Ansatz 2 sollte bessere Generalisierung zeigen.

---

### 3.2 Experiment 2: Architektur-Vergleich

**Ziel:** Einfluss der Anzahl verdeckter Schichten auf die Klassifikationsleistung untersuchen.

**Getestete Konfigurationen:**
1. **1 verdeckte Schicht** - Einfaches Modell
2. **3 verdeckte Schichten** - Mittlere Komplexität
3. **5 verdeckte Schichten** - Komplexes Modell

**Jede Schicht:**
- 64 Neuronen
- ReLU-Aktivierung
- 30% Dropout

**Hypothese:**
- Zu wenige Schichten → Underfitting (kann Muster nicht erfassen)
- Zu viele Schichten → Overfitting (lernt Trainingsdaten auswendig)
- Optimale Balance führt zu bester Testgenauigkeit

---

## 4. Implementierung

### 4.1 Dateneinlesung

```python
class DataLoader:
    def load_images(self):
        """Lädt Bilder auf Pixelbasis"""
        for img_file in all_files:
            # Lade Bild in Graustufen
            img = image.load_img(img_path, 
                               target_size=IMG_SIZE, 
                               color_mode='grayscale')
            
            # Konvertiere zu Array
            img_array = image.img_to_array(img)
            
            # Normalisiere Pixelwerte
            img_array = img_array / 255.0
```

### 4.2 Modelltraining

```python
# Kompilierung
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=1,
    validation_data=(X_test, y_test)
)
```

### 4.3 Evaluation

```python
# Testgenauigkeit
test_loss, test_acc = model.evaluate(X_test, y_test)

# Vorhersagen
predictions = model.predict(X_test)
```

---

## 5. Ergebnisse

### 5.1 Trainingsansätze

| Ansatz | Test Accuracy | Beschreibung |
|--------|--------------|--------------|
| Ansatz 1 (Sequentiell) | XX.XX% | Trainiert Symbole nacheinander |
| Ansatz 2 (Gemischt) | XX.XX% | Trainiert Symbole durchmischt |

**Beobachtungen:**
- [Hier deine Ergebnisse eintragen nach Ausführung]
- Ansatz 2 zeigt voraussichtlich bessere Generalisierung
- Ansatz 1 könnte unter Catastrophic Forgetting leiden

### 5.2 Architektur-Vergleich

| Anzahl Hidden Layers | Test Accuracy | Test Loss |
|---------------------|---------------|-----------|
| 1 | XX.XX% | X.XXXX |
| 3 | XX.XX% | X.XXXX |
| 5 | XX.XX% | X.XXXX |

**Beobachtungen:**
- [Hier deine Ergebnisse eintragen]
- Optimale Architektur: [X] Schichten
- Trade-off zwischen Modellkomplexität und Generalisierung

### 5.3 Visualisierungen

Die generierten Plots zeigen:
1. **predictions.png**: Beispielvorhersagen mit Konfidenz
2. **architecture_comparison.png**: Vergleich der Architekturen

---

## 6. Diskussion

### 6.1 Eigenschaften neuronaler Netze (Experiment 1)

**Catastrophic Forgetting:**
- Bei sequentiellem Training (Ansatz 1) kann das Netzwerk zuvor gelernte Informationen "vergessen"
- Das Netzwerk passt seine Gewichte stark an die aktuellen Daten an
- Früher gelernte Muster werden überschrieben

**Stochastisches Training:**
- Bei gemischtem Training (Ansatz 2) lernt das Netzwerk beide Klassen gleichzeitig
- Gewichte werden ausbalanciert für beide Symbole
- Bessere Generalisierung auf unbekannte Daten

**Praktische Implikation:**
In realen Anwendungen sollten Trainingsdaten immer gut durchmischt werden, um Catastrophic Forgetting zu vermeiden.

### 6.2 Modellkomplexität (Experiment 2)

**Underfitting (zu wenige Schichten):**
- Modell ist zu einfach
- Kann komplexe Muster nicht erfassen
- Schlechte Performance auf Train- und Testdaten

**Overfitting (zu viele Schichten):**
- Modell ist zu komplex
- Lernt Trainingsdaten auswendig (inkl. Rauschen)
- Gute Performance auf Trainingsdaten, schlecht auf Testdaten

**Optimale Komplexität:**
- Balance zwischen Lernfähigkeit und Generalisierung
- Wird durch Validierung auf Testdaten ermittelt
- Regularisierungstechniken (Dropout) helfen gegen Overfitting

### 6.3 Rolle der CNNs

**Warum CNNs für Bilderkennung?**
- **Lokale Konnektivität**: Neuronen sind nur mit lokalen Regionen verbunden
- **Parameter-Sharing**: Gleiche Filter werden auf gesamtes Bild angewendet
- **Hierarchische Features**: Frühe Schichten erkennen einfache Features (Kanten), tiefe Schichten komplexe Muster

---

## 7. Schlussfolgerung

### 7.1 Wichtigste Erkenntnisse

1. **Gemischtes Training** ist dem sequentiellen Training überlegen und verhindert Catastrophic Forgetting
2. Die **Anzahl verdeckter Schichten** beeinflusst die Modellperformance erheblich
3. **CNNs** sind hocheffektiv für Bildklassifikation durch ihre spezialisierte Architektur
4. **Regularisierung** (Dropout) ist wichtig zur Vermeidung von Overfitting

### 7.2 Anwendung

Die entwickelten Modelle können:
- Handgeschriebene Mathematiksymbole mit hoher Genauigkeit klassifizieren
- Als Basis für umfangreichere Symbol-Erkennungssysteme dienen
- Für automatische Korrektur mathematischer Aufgaben eingesetzt werden

---

## 8. Quellen

### 8.1 Datensatz
- Kaggle Dataset: [Handwritten Math Symbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)

### 8.2 Literatur
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Chollet, F. (2021). *Deep Learning with Python*. Manning Publications.
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

### 8.3 Online-Ressourcen
- [PyTorch Tutorial](https://pytorch.org/tutorials/)
- [Introduction to TensorFlow](https://www.tensorflow.org/tutorials)
- [Introduction to Deep Learning in Python (DataCamp)](https://www.datacamp.com/)

### 8.4 Code-Referenzen
```python
# Beispiel: Bildeinlesung mit TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing import image

img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
```

---

## 9. Anhang

### 9.1 Vollständiger Code

Der vollständige Code ist in `main.py` verfügbar.

### 9.2 Ausführung

```bash
# Installation der Abhängigkeiten
pip install -r requirements.txt

# Ausführung des Programms
python main.py
```

### 9.3 Ausgabedateien

- `predictions.png`: Visualisierung der Vorhersagen
- `architecture_comparison.png`: Vergleich der Architekturen
- Konsolenoutput mit detaillierten Metriken

