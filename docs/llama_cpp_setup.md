# Installationsanleitung für llama.cpp auf dem Ubuntu-Server

Diese Anleitung beschreibt die Installation und Einrichtung von llama.cpp für die Ausführung des trainierten DeepSeek-Modells.

## Voraussetzungen

- Ubuntu Server 24.04.2 LTS
- Mindestens 8GB RAM (16GB+ empfohlen)
- Freier Speicherplatz (mindestens 10GB)
- Grundlegende Linux-Kenntnisse

## 1. Abhängigkeiten installieren

```bash
# System-Pakete aktualisieren
sudo apt update
sudo apt upgrade -y

# Entwicklungstools installieren
sudo apt install -y build-essential cmake git python3-dev python3-pip

# Optional: GPU-Abhängigkeiten installieren (falls NVIDIA GPU vorhanden)
# CUDA-Toolkit installieren (Version 12.2)
sudo apt install -y nvidia-driver-535
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --toolkit --silent

# Umgebungsvariablen für CUDA einrichten
echo 'export PATH=$PATH:/usr/local/cuda-12.2/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64' >> ~/.bashrc
source ~/.bashrc