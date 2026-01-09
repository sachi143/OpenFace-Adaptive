import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "demo_log.csv"

def plot_trust():
    if not os.path.exists(LOG_FILE):
        print(f"No log file found at {LOG_FILE}. Run live_demo.py first!")
        return
        
    # Read Data (skip header row)
    df = pd.read_csv(LOG_FILE, skiprows=1, names=['timestamp', 'emotion', 'v_score', 'a_score'])
    
    if len(df) == 0:
        print("No data in log file. Run live_demo.py for at least a few seconds first!")
        return
    
    # Normalize Time
    start_time = df['timestamp'].iloc[0]
    df['time_rel'] = df['timestamp'] - start_time
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['time_rel'], df['v_score'], label='Visual Trust', color='blue', linewidth=2)
    plt.plot(df['time_rel'], df['a_score'], label='Audio Trust', color='red', linewidth=2, linestyle='--')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Reliability Score (0-1)')
    plt.title('Explainability: Reliability Gates over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Annotate Emotions as scatter points or changing background?
    # Let's just annotate changes
    # Simple Scatter
    # plt.scatter(df['time_rel'], [1.05]*len(df), c='gray', marker='|') 
    
    plt.savefig('figure_trust_scores.png', dpi=300)
    print("Saved plot to figure_trust_scores.png")
    plt.show()

if __name__ == "__main__":
    plot_trust()
