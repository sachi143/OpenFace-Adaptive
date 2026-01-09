import torch
import torch.quantization
from model import OpenFaceAdaptiveNet
import os

MODEL_PATH = "openface_adaptive_v1.pth"
QUANT_PATH = "openface_adaptive_quantized.pth"

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def main():
    print("Loading Baseline Model...")
    model = OpenFaceAdaptiveNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    print("Baseline Model Size:")
    print_size_of_model(model)
    
    print("\nQuantizing (Dynamic INT8)...")
    # Quantize Linear, LSTM, RNN, Transformer layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, torch.nn.TransformerEncoderLayer}, 
        dtype=torch.qint8
    )
    
    print("Quantized Model Size:")
    print_size_of_model(quantized_model)
    
    print(f"\nSaving to {QUANT_PATH}...")
    torch.save(quantized_model.state_dict(), QUANT_PATH)
    print("Done. Use this model on Raspberry Pi!")

if __name__ == "__main__":
    main()
