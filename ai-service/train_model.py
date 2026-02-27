"""
=============================================================
POLYGLOT GHOST — Deep Learning Training Script (Run on Kaggle)
=============================================================

TRAINS a 1D CNN on mel-spectrograms for audio deepfake detection.
Exports to ONNX format for lightweight inference (~50MB onnxruntime).

INSTRUCTIONS:
1. Create a new Kaggle Notebook (with GPU accelerator!)
2. Add your FoR (Fake or Real) dataset
3. Paste this ENTIRE script into a code cell
4. Update DATASET_PATH on line 38
5. Run it (~15-30 minutes with GPU)
6. Download from Output tab:
   - deepfake_cnn.onnx  (the trained model)
7. Put it in your local: ai-service/model/deepfake_cnn.onnx
8. Restart: python app.py

=============================================================
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURE — UPDATE THIS PATH!
# ============================================================
DATASET_PATH = "/kaggle/input/fake-or-real-for-dataset"  # <-- CHANGE THIS

# To find the correct path, run this first:
# import os; print(os.listdir("/kaggle/input/"))

MAX_FILES_PER_CLASS = 3000   # More = better accuracy, slower training
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
TARGET_SR = 16000
MAX_DURATION = 3  # seconds
N_MELS = 64       # mel spectrogram bands
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_PATH}")


# ============================================================
# FIND AUDIO FILES
# ============================================================

def find_audio_files(base_path):
    """Auto-detect dataset structure and find real/fake audio files."""
    real_files = []
    fake_files = []

    patterns = [
        ("real", "fake"), ("REAL", "FAKE"), ("Real", "Fake"),
        ("genuine", "spoof"), ("bonafide", "spoof"),
        ("training/real", "training/fake"),
        ("train/real", "train/fake"),
        ("for-original", "for-rerec"),  # FoR dataset structure
        ("for-norm/training/real", "for-norm/training/fake"),
        ("for-2sec/training/real", "for-2sec/training/fake"),
    ]

    for real_name, fake_name in patterns:
        real_dir = os.path.join(base_path, real_name)
        fake_dir = os.path.join(base_path, fake_name)

        if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
            print(f"✅ Found: {real_name}/ and {fake_name}/")
            for f in os.listdir(real_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    real_files.append(os.path.join(real_dir, f))
            for f in os.listdir(fake_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    fake_files.append(os.path.join(fake_dir, f))
            break

    if not real_files or not fake_files:
        print("Standard folders not found. Recursive search...")
        for root, dirs, files in os.walk(base_path):
            folder = os.path.basename(root).lower()
            for f in files:
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    path = os.path.join(root, f)
                    if any(x in folder for x in ['fake', 'spoof', 'rerec', 'synth']):
                        fake_files.append(path)
                    elif any(x in folder for x in ['real', 'genuine', 'bonafide', 'original']):
                        real_files.append(path)

    return real_files, fake_files


# ============================================================
# MEL SPECTROGRAM EXTRACTION
# ============================================================

def audio_to_melspec(file_path, sr=TARGET_SR, duration=MAX_DURATION, n_mels=N_MELS):
    """Convert audio file to fixed-size mel spectrogram."""
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception:
        return None

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Pad or truncate to fixed length
    target_len = sr * duration
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr//2)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_db.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = torch.FloatTensor(self.specs[idx]).unsqueeze(0)  # Add channel dim
        label = torch.LongTensor([self.labels[idx]])[0]
        return spec, label


# ============================================================
# CNN MODEL
# ============================================================

class DeepfakeCNN(nn.Module):
    """1D CNN for mel spectrogram classification.
    Input: (batch, 1, n_mels, time_frames)
    Output: (batch, 2) — probabilities for [REAL, FAKE]
    """
    def __init__(self, n_mels=N_MELS):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# MAIN TRAINING
# ============================================================

print("\n" + "=" * 60)
print("POLYGLOT GHOST — Deep Learning Training")
print("=" * 60)

# Step 1: Find files
print(f"\n📂 Scanning dataset...")
real_files, fake_files = find_audio_files(DATASET_PATH)
print(f"  REAL: {len(real_files)} files")
print(f"  FAKE: {len(fake_files)} files")

if not real_files or not fake_files:
    print("\n❌ ERROR: Could not find audio files!")
    if os.path.exists(DATASET_PATH):
        print(f"\nContents of {DATASET_PATH}:")
        for item in sorted(os.listdir(DATASET_PATH))[:30]:
            full = os.path.join(DATASET_PATH, item)
            if os.path.isdir(full):
                count = len(os.listdir(full))
                print(f"  📁 {item}/ ({count} items)")
            else:
                print(f"  📄 {item}")
    raise SystemExit("Fix DATASET_PATH and re-run")

# Limit files
np.random.seed(42)
if len(real_files) > MAX_FILES_PER_CLASS:
    real_files = list(np.random.choice(real_files, MAX_FILES_PER_CLASS, replace=False))
if len(fake_files) > MAX_FILES_PER_CLASS:
    fake_files = list(np.random.choice(fake_files, MAX_FILES_PER_CLASS, replace=False))

print(f"  Using: {len(real_files)} REAL + {len(fake_files)} FAKE")

# Step 2: Extract mel spectrograms
print(f"\n🎵 Extracting mel spectrograms...")

specs = []
labels = []
errors = 0

for fpath in tqdm(real_files, desc="  REAL"):
    mel = audio_to_melspec(fpath)
    if mel is not None:
        specs.append(mel)
        labels.append(0)  # Class 0 = REAL
    else:
        errors += 1

for fpath in tqdm(fake_files, desc="  FAKE"):
    mel = audio_to_melspec(fpath)
    if mel is not None:
        specs.append(mel)
        labels.append(1)  # Class 1 = FAKE
    else:
        errors += 1

specs = np.array(specs)
labels = np.array(labels)

print(f"\n  Total: {len(specs)} spectrograms ({errors} errors)")
print(f"  Shape: {specs.shape}")
print(f"  REAL: {np.sum(labels==0)}, FAKE: {np.sum(labels==1)}")

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    specs, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# Step 4: Train
print(f"\n🧠 Training CNN on {DEVICE}...")

model = DeepfakeCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_acc = 0
best_state = None

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for specs_batch, labels_batch in train_loader:
        specs_batch = specs_batch.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(specs_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()

    train_acc = correct / total

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for specs_batch, labels_batch in test_loader:
            specs_batch = specs_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            outputs = model(specs_batch)
            _, predicted = outputs.max(1)
            test_total += labels_batch.size(0)
            test_correct += predicted.eq(labels_batch).sum().item()

    test_acc = test_correct / test_total
    scheduler.step(1 - test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        best_state = model.state_dict().copy()
        marker = " ★ BEST"
    else:
        marker = ""

    if (epoch + 1) % 5 == 0 or marker:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss={train_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}{marker}")

# Load best model
model.load_state_dict(best_state)
print(f"\n  Best test accuracy: {best_acc:.4f}")

# Final evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for specs_batch, labels_batch in test_loader:
        specs_batch = specs_batch.to(DEVICE)
        outputs = model(specs_batch)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

print("\n📊 Final Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['REAL', 'FAKE']))

# Step 5: Export to ONNX
print("📦 Exporting to ONNX format...")

model.cpu()
model.eval()

# Get the correct input shape from our data
sample_shape = specs[0].shape  # (n_mels, time_frames)
dummy_input = torch.randn(1, 1, sample_shape[0], sample_shape[1])

onnx_path = "/kaggle/working/deepfake_cnn.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['audio_spectrogram'],
    output_names=['prediction'],
    dynamic_axes={
        'audio_spectrogram': {0: 'batch_size'},
        'prediction': {0: 'batch_size'}
    },
    opset_version=11,
)

# Save the spectrogram shape info
import json
config = {
    "n_mels": N_MELS,
    "sr": TARGET_SR,
    "max_duration": MAX_DURATION,
    "input_shape": list(sample_shape),
    "classes": {0: "REAL", 1: "FAKE"},
    "best_accuracy": float(best_acc),
}
with open("/kaggle/working/model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"  ✅ {onnx_path} saved")
print(f"  ✅ model_config.json saved")

# Check file size
size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"  Model size: {size_mb:.1f} MB")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print(f"   Best Accuracy: {best_acc:.2%}")
print(f"   Model size:    {size_mb:.1f} MB")
print(f"   Class 0 = REAL")
print(f"   Class 1 = FAKE")
print("=" * 60)
print("\n📥 DOWNLOAD FROM OUTPUT TAB:")
print("  1. deepfake_cnn.onnx")
print("  2. model_config.json")
print("\n📁 PUT THEM IN: ai-service/model/")
print("🔄 RESTART: python app.py")
