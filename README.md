# Gesture-Controlled PDF Viewer

A Python-based project that allows users to navigate PDF files using **hand gestures, head movements, swipe actions, and voice commands**.  
Built with **OpenCV**, **MediaPipe**, and **PyMuPDF**, this project provides an innovative way to control documents without using a keyboard or mouse.

---

## ✨ Features

- 📄 **PDF Viewing** – Load and display PDF files.
- ✋ **Hand Gesture Controls** – Navigate pages using finger gestures.
- 🧑‍🦱 **Head Gesture Scrolling** – Tilt head to scroll pages.
- ↔️ **Swipe Gestures** – Swipe left/right to change pages quickly.
- 🎙️ **Voice Commands** – Navigate or jump to specific pages using voice input.
- 🔍 **Zoom Support** – Pinch gesture to zoom in/out.
- 💡 **UI Overlay** – On-screen hints for available commands.

---

## 🛠️ Tech Stack

- **Python 3.x**
- [OpenCV](https://opencv.org/) – Computer vision
- [MediaPipe](https://developers.google.com/mediapipe) – Hand & face tracking
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) – PDF rendering
- [NumPy](https://numpy.org/) – Numerical operations
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) – Voice commands

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abhirajbhat/Gesture-pdf.git
   cd gesture-pdf-viewer
2. Create a virtual environment (recommended):
     python -m venv .venv
    source .venv/bin/activate   # For Linux/Mac
    .venv\Scripts\activate      # For Windows
3. Install dependencies:
    pip install -r requirements.txt

## ▶️ Usage
    1.Place your PDF file (e.g., Tutorial_EDIT.pdf) in the project folder.
    2.Run the script:
        python gesture.py
    3.Use gestures, swipes, head tilts, or voice commands to navigate your PDF.


## 🎮 Controls

**Swipe Left/Right → Move between pages**
**Finger Gestures → Scroll up/down**
**Head Tilt → Smooth scrolling**
**Pinch Zoom → Zoom in/out**
**Voice Command → e.g., "Go to page 5"**

## 📂 Project Structure
    gesture-pdf-viewer/
    │── gesture.py          # Main script
    │── requirements.txt    # Dependencies
    │── README.md           # Documentation
    │── .venv/              # Virtual environment (optional)


## 📜 License
 This project is licensed under the MIT License – see the LICENSE
  file for details.

## 👨‍💻 Author
Abhiraj Bhat,Aspiring Data Scientist
