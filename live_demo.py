import subprocess
import torch
import numpy as np
import pyaudio
import librosa
import threading
import queue
import time
import os
import pandas as pd
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModel
from model import OpenFaceAdaptiveNet

# ==========================================
# CONFIGURATION
# ==========================================
# PATH TO OPENFACE 
OPENFACE_PATH = r"./OpenFace_2.2.0/FeatureExtraction.exe" 

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2 # Window size for inference

# Model Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. SHARED STATE
# ==========================================
# Queues to hold the latest features from each thread
visual_queue = queue.Queue(maxsize=1)
audio_queue = queue.Queue(maxsize=1)
text_queue = queue.Queue(maxsize=1)

# Global flag to stop threads
running = True

# ==========================================
# 2. SENSOR THREADS
# ==========================================

def audio_worker():
    """
    Continuously records audio and extracts MFCCs.
    """
    print("[Audio] Thread Started")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    while running:
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        # Convert to numpy
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
        
        # Extract MFCCs (74 dims to match model)
        # We extract 13 MFCCs + deltas + delta-deltas usually, but here we force 74 for the model
        # For simplicity in this demo, we'll extract 74 melspectrogram bins or pad MFCCs
        mfcc = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=74)
        mfcc_avg = np.mean(mfcc, axis=1) # Average over time -> (74,)
        
        # Push to queue
        tensor = torch.tensor(mfcc_avg).unsqueeze(0) # (1, 74)
        if audio_queue.full(): audio_queue.get()
        audio_queue.put(tensor)
        
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("[Audio] Thread Stopped")

def video_worker():
    """
    Runs OpenFace FeatureExtraction in a subprocess and reads the CSV output from stdout.
    """
    print("[Video] Thread Started")
    
    if not os.path.exists(OPENFACE_PATH):
        print(f"[Video] ERROR: OpenFace not found at {OPENFACE_PATH}")
        return

    # Command to run OpenFace on webcam (device 0)
    # We use -out_dir to save CSV to a known location
    # -force to overwrite existing files
    out_dir = "temp_out"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    cmd = [OPENFACE_PATH, "-device", "0", "-aus", "-pose", "-gaze", "-out_dir", out_dir, "-of", "webcam_out", "-force"]
    
    print(f"[Video] Launching: {' '.join(cmd)}")
    try:
        # Start OpenFace in background
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True  # Capture as string
        )
        
        # Give it a second to create the file
        time.sleep(3.0)
        
        # Check if it died
        if process.poll() is not None:
            out, err = process.communicate()
            print(f"[Video] ERROR: OpenFace exited immediately (Code {process.returncode})")
            print(f"STDOUT: {out}")
            print(f"STDERR: {err}")
            return

        csv_path = os.path.join(out_dir, "webcam_out.csv")
        
        # Wait longer for CSV to be created
        max_wait = 10
        for i in range(max_wait):
            if os.path.exists(csv_path):
                break
            print(f"[Video] Waiting for OpenFace CSV... ({i+1}/{max_wait}s)")
            time.sleep(1)
        
        if not os.path.exists(csv_path):
            # Check stderr for errors
            if process.poll() is not None:
                _, err = process.communicate()
                print(f"[Video] OpenFace STDERR: {err}")
            print(f"[Video] ERROR: CSV not created. Check if camera is accessible.")
            print(f"[Video] TIP: Make sure no other app is using the webcam.")
            return
        else:
            print(f"[Video] Reading from {csv_path}...")
        
        # Open file for reading
        with open(csv_path, 'r') as f:
            # Skip header
            header = f.readline()
            while running:
                # Read new line
                line = f.readline()
                if not line:
                    time.sleep(0.05) # Wait for new data
                    continue
                
                # Parse Line
                parts = line.strip().split(',')
                if len(parts) < 10: continue # Invalid line
                
                # Extract Features
                # OpenFace CSV structure varies but generally:
                # frame, face_id, timestamp, ... [Features] ...
                # We need to extract the relevant columns. 
                # For this demo, we take columns 4 onwards (skipping meta) to fill our vector
                # This is a HEURISTIC mapping. Ideally we map by header name.
                
                try:
                    # Parse all floats
                    vals = [float(x) for x in parts[4:]] # Skip frame, id, timestamp, confidence
                    
                    # Create Tensor (1, 713)
                    feats = torch.zeros(1, 713)
                    
                    # Fill available features (usually ~40-50 from -aus -pose -gaze)
                    # We map them to the start of the 713 vector. 
                    # The model will use what it can.
                    length = min(len(vals), 713)
                    feats[0, :length] = torch.tensor(vals[:length])
                    
                    if visual_queue.full(): visual_queue.get()
                    visual_queue.put(feats)
                    
                except ValueError:
                    continue

    except Exception as e:
        print(f"[Video] Error: {e}")
    finally:
        # Cleanup
        if 'process' in locals(): process.terminate()
             
    print("[Video] Thread Stopped")

def text_worker():
    """
    Uses SpeechRecognition to listen and transcribe.
    """
    print("[Text] Thread Started")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased")
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        
    while running:
        try:
            with mic as source:
                # Listen for a short phrase
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            text = recognizer.recognize_google(audio)
            print(f"[Text] Heard: '{text}'")
            
            # Tokenize and Embed
            inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=20, truncation=True)
            with torch.no_grad():
                outputs = bert(**inputs)
            
            # Use CLS token embedding (1, 768)
            emb = outputs.last_hidden_state[:, 0, :]
            
            if text_queue.full(): text_queue.get()
            text_queue.put(emb)
            
        except sr.WaitTimeoutError:
            pass # No speech
        except sr.UnknownValueError:
            pass # Unintelligible
        except Exception as e:
            print(f"[Text] Error: {e}")

    print("[Text] Thread Stopped")

# ==========================================
# 3. MAIN INFERENCE LOOP
# ==========================================
def get_live_inference():
    global running
    
    # 1. Load Model (try different hidden_dim to match checkpoint)
    print("Loading Model...")
    
    model_paths = ["results/model_baseline.pth", "openface_adaptive_v1.pth", "openface_adaptive_quantized.pth"]
    hidden_dims_to_try = [256, 192, 128]  # Try different dims
    loaded = False
    
    for path in model_paths:
        if os.path.exists(path):
            for hdim in hidden_dims_to_try:
                try:
                    model = OpenFaceAdaptiveNet(hidden_dim=hdim)
                    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                    print(f"Model loaded from: {path} (hidden_dim={hdim})")
                    loaded = True
                    break
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        continue
                    else:
                        raise e
            if loaded:
                break
    
    if not loaded:
        print("WARNING: No compatible model found! Using random weights with hidden_dim=192.")
        model = OpenFaceAdaptiveNet(hidden_dim=192)
    
    model.eval()
    
    # 2. Start Threads (daemon=True for clean exit)
    t_audio = threading.Thread(target=audio_worker, daemon=True)
    t_video = threading.Thread(target=video_worker, daemon=True)
    t_text = threading.Thread(target=text_worker, daemon=True)
    
    t_audio.start()
    t_video.start()
    t_text.start()
    
    print("\nSystem Active. Press Ctrl+C to stop.\n")
    
    # Initialize Log
    log_path = os.path.abspath("demo_log.csv")
    with open(log_path, "w") as f:
        f.write("timestamp,emotion,v_score,a_score\n")
    print(f"[Log] Writing trust scores to: {log_path}")

    
    try:
        while True:
            # Get latest features (Non-blocking or with timeout)
            # We need all three to make a prediction
            
            # Default to zeros if queue is empty (e.g. silence, no face)
            v_feat = visual_queue.get() if not visual_queue.empty() else torch.zeros(1, 713)
            a_feat = audio_queue.get() if not audio_queue.empty() else torch.zeros(1, 74)
            t_feat = text_queue.get() if not text_queue.empty() else torch.zeros(1, 768)
            
            # Ensure correct shape
            if v_feat.dim() == 1:
                v_feat = v_feat.unsqueeze(0)  # [713] -> [1, 713]
            if v_feat.shape[1] < 713:
                # Pad if less than 713
                v_feat_padded = torch.zeros(1, 713)
                v_feat_padded[:, :v_feat.shape[1]] = v_feat
                v_feat = v_feat_padded
            elif v_feat.shape[1] > 713:
                v_feat = v_feat[:, :713]  # Trim
            
            # 2. Text: Slice 768 -> 300 (DistilBERT -> GloVe dimension)
            t_feat_sliced = t_feat[:, :300]
            
            # Inference
            with torch.no_grad():
                prediction, v_score, a_score = model(v_feat, a_feat, t_feat_sliced)
                emotion_idx = torch.argmax(prediction).item()
                
            emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
            
            # XAI LOGGING
            timestamp = time.time()
            with open("demo_log.csv", "a") as f:
                f.write(f"{timestamp},{emotions[emotion_idx]},{v_score.item()},{a_score.item()}\n")
            
            print(f"\rEmotion: {emotions[emotion_idx]:<10} | Visual Trust: {v_score.item():.2f} | Audio Trust: {a_score.item():.2f}", end="")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping... (Ctrl+C received)")
        running = False
        print("Cleaning up threads...")
        # Give threads time to exit gracefully
        time.sleep(1)
        
        # Force kill OpenFace subprocess (Windows)
        print("Killing OpenFace process...")
        os.system("taskkill /IM FeatureExtraction.exe /F 2>nul")
        
        print("Done! Check demo_log.csv for trust scores.")
        print("Run 'python plot_trust.py' to generate the figure.")

if __name__ == "__main__":
    get_live_inference()