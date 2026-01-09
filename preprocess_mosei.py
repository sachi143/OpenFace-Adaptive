import h5py
import numpy as np
import pickle
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DIR = r"c:\Openface-Adaptive\cmu_mosei_raw\CMU-MOSEI"
OUTPUT_FILE = "mosei_data.pkl"

# File Paths
LABELS_PATH = os.path.join(RAW_DIR, "labels", "CMU_MOSEI_Labels.csd")
VISUAL_PATH = os.path.join(RAW_DIR, "visuals", "CMU_MOSEI_VisualOpenFace2.csd")
AUDIO_PATH  = os.path.join(RAW_DIR, "acoustics", "CMU_MOSEI_COVAREP.csd")
TEXT_PATH   = os.path.join(RAW_DIR, "languages", "CMU_MOSEI_TimestampedWordVectors.csd") # Glove/BERT

def get_data_dict(file_path):
    """
    Opens HDF5 file and returns the 'data' group which contains VideoIDs.
    Shape: Root -> DatasetName -> data -> VideoID
    """
    f = h5py.File(file_path, 'r')
    # Dynamic root key access
    root_key = list(f.keys())[0] 
    return f[root_key]['data'], f

def get_avg_features(features, intervals, start_time, end_time):
    """
    Average pooling of features within the time interval [start_time, end_time].
    """
    if features.shape[0] == 0: return np.zeros(features.shape[1])
    
    # 1. Find indices where intervals overlap with our target segment
    # Intervals shape: (N, 2)
    # simple overlap check: (int_start < end_time) and (int_end > start_time)
    
    # Vectorized search is faster but let's do simple boolean masking
    mask = (intervals[:, 0] < end_time) & (intervals[:, 1] > start_time)
    
    selected_feats = features[mask]
    
    if selected_feats.shape[0] == 0:
        return np.zeros(features.shape[1])
    
    # Feature Engineering: Clean Data
    # Replace Inf with 0 (Common in audio features like log-energy)
    # Replace NaN with 0
    selected_feats = np.nan_to_num(selected_feats, posinf=0.0, neginf=0.0, nan=0.0)
    
    return np.mean(selected_feats, axis=0)

def main():
    print("Opening CSD files...")
    
    # Keep files open
    l_data, f_l = get_data_dict(LABELS_PATH)
    v_data, f_v = get_data_dict(VISUAL_PATH)
    a_data, f_a = get_data_dict(AUDIO_PATH)
    t_data, f_t = get_data_dict(TEXT_PATH)
    
    processed_data = {
        'train': {'visual': [], 'audio': [], 'text': [], 'labels': []},
        'valid': {'visual': [], 'audio': [], 'text': [], 'labels': []},
        'test': {'visual': [], 'audio': [], 'text': [], 'labels': []}
    }
    
    print("Processing Segments...")
    # Iterate over all videos in Labels
    # Use tqdm for progress
    video_ids = list(l_data.keys())
    
    valid_segments = 0
    missing_modality_count = 0
    
    for vid in tqdm(video_ids):
        # Decode if bytes
        vid_str = vid
        
        # Check if this video exists in all other modalities
        if (vid_str not in v_data) or (vid_str not in a_data) or (vid_str not in t_data):
            missing_modality_count += 1
            continue
            
        # Get Label Segments
        # Labels are small, load all
        l_feats = l_data[vid_str]['features'][:]
        l_ints  = l_data[vid_str]['intervals'][:]
        
        # GET HANDLES ONLY (Lazy Load)
        v_dset = v_data[vid_str]['features']
        v_ints = v_data[vid_str]['intervals'][:] # Load intervals (small) to find indices
        
        a_dset = a_data[vid_str]['features']
        a_ints = a_data[vid_str]['intervals'][:]
        
        t_dset = t_data[vid_str]['languages'] if 'languages' in t_data[vid_str] else t_data[vid_str]['features']
        # Note: Text structure might vary, but assuming keys
        t_ints = t_data[vid_str]['intervals'][:]
        
        # Determine Split
        h = hash(vid_str) % 10
        if h < 7: split = 'train'
        elif h < 8: split = 'valid'
        else: split = 'test'
        
        # Iterate over each segment in this video
        for i in range(l_feats.shape[0]):
            score = l_feats[i][0]
            start = l_ints[i][0]
            end   = l_ints[i][1]
            
            # Helper to extract slice
            def load_slice(dset, intervals, s, e):
                # Find range of indices where intervals overlap with s, e
                # intervals is (N, 2)
                # We want indices where (start < e) & (end > s)
                # Since intervals are usually chronological, we can find first and last index
                
                # Optimized search:
                # valid_indices = np.where((intervals[:,0] < e) & (intervals[:,1] > s))[0]
                # if len(valid_indices) == 0: return np.zeros(dset.shape[1])
                # slice_start = valid_indices[0]
                # slice_end = valid_indices[-1] + 1
                
                # Slicing HDF5
                # data = dset[slice_start:slice_end]
                
                # But 'dset' is HDF5 dataset. Standard slicing works.
                # However, boolean mask doesn't work well on Dataset.
                # So we must compute start/end integers.
                
                mask = (intervals[:, 0] < e) & (intervals[:, 1] > s)
                if not mask.any(): return np.zeros(dset.shape[1])
                
                indices = np.where(mask)[0]
                s_idx, e_idx = indices[0], indices[-1] + 1
                
                # Read valid chunk
                raw = dset[s_idx:e_idx]
                
                # Additional: clean
                raw = np.nan_to_num(raw, posinf=0.0, neginf=0.0, nan=0.0)
                return np.mean(raw, axis=0)

            # Extract aligned features
            v_vec = load_slice(v_dset, v_ints, start, end)
            a_vec = load_slice(a_dset, a_ints, start, end)
            
            # Text might be tricky if it's 'languages' group? 
            # Assuming t_dset is dataset for now based on previous manual check
            t_vec = load_slice(t_dset, t_ints, start, end)
            
            # Map Label
            label_class = max(0, min(6, int(round(score) + 3)))
            
            # Append
            processed_data[split]['visual'].append(v_vec.astype(np.float32))
            processed_data[split]['audio'].append(a_vec.astype(np.float32))
            processed_data[split]['text'].append(t_vec.astype(np.float32))
            processed_data[split]['labels'].append(label_class)
            
            valid_segments += 1

    print("\nProcessing Complete.")
    print(f"Total Valid Segments: {valid_segments}")
    print(f"Videos skipped due to missing modalities: {missing_modality_count}")
    
    # Convert lists to arrays
    for split in processed_data:
        print(f"Finalizing {split} split...")
        for k in processed_data[split]:
            processed_data[split][k] = np.array(processed_data[split][k])
            print(f"  {k}: {processed_data[split][k].shape}")
            
    print(f"Saving to {OUTPUT_FILE} (This might take a moment)...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(processed_data, f)
        
    print("Done! Ready for training.")
    
    # Close files
    f_l.close()
    f_v.close()
    f_a.close()
    f_t.close()

if __name__ == "__main__":
    main()
