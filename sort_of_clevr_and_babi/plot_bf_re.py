import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Configuration of file names (Complete attention before Top-K)
file_hx1 = "hx1_scores_bf.npy"
file_hx2 = "hx2_scores_bf.npy"

# 2. Set visualization hyperparameters
HEAD_AVG = False    # False means do not average, only view a specific attention head
HEAD_IDX = 4        # If HEAD_AVG is False, view the 0th head

def plot_all_batches_heatmaps():
    # Check if files exist
    if not (os.path.exists(file_hx1) and os.path.exists(file_hx2)):
        print("Cannot find the specified .npy files. Please make sure hx1_scores_bf.npy and hx2_scores_bf.npy are in the current directory.")
        return
        
    data_hx1 = np.load(file_hx1)
    data_hx2 = np.load(file_hx2)
    
    num_layers = data_hx1.shape[0]  # Expected to be 8 layers
    batch_size = data_hx1.shape[1]  # Expected to be 64 samples
    
    # ================= Create a directory for batch images =================
    save_dir = "plot_all_bf_re"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get head string for naming
    if HEAD_AVG:
        head_str = "HeadAvg"
    else:
        head_str = f"Head{HEAD_IDX}"
        
    print(f"Starting to generate attention heatmaps for {batch_size} samples...\n")

    # ================= Iterate through all batch samples =================
    for batch_idx in range(batch_size):
        print(f"Processing Sample {batch_idx + 1}/{batch_size} ...")
        
        # Create a canvas with 2 rows and num_layers columns
        fig, axes = plt.subplots(2, num_layers, figsize=(32, 10))
        
        for layer_idx in range(num_layers):
            # Get data for the current layer and batch
            rev_layer_idx = num_layers - 1 - layer_idx
        
            sample_hx1 = data_hx1[layer_idx, batch_idx]
            sample_hx2 = data_hx2[rev_layer_idx, batch_idx]
            
            # Extract attention head information
            if HEAD_AVG:
                plot_hx1 = np.mean(sample_hx1, axis=0)
                plot_hx2 = np.mean(sample_hx2, axis=0)
                title_suffix = "Avg Heads"
            else:
                plot_hx1 = sample_hx1[HEAD_IDX]
                plot_hx2 = sample_hx2[HEAD_IDX]
                title_suffix = f"Head {HEAD_IDX}"
                
            # --- First row: Working Memory (hx1) ---
            ax1 = axes[0, layer_idx]
            sns.heatmap(plot_hx1, ax=ax1, cmap="YlGnBu", cbar=(layer_idx == num_layers - 1)) 
            ax1.set_title(f"Layer {layer_idx + 1}\nWorking Memory", fontsize=14)
            ax1.set_xlabel("Memory Slots", fontsize=12)
            
            if layer_idx == 0:
                ax1.set_ylabel("Input Sequence (0-13)", fontsize=14)
            else:
                ax1.set_ylabel("")
            ax1.tick_params(axis='y', rotation=0)
            
            # --- Second row: Long-term Memory (hx2) ---
            ax2 = axes[1, layer_idx]
            sns.heatmap(plot_hx2, ax=ax2, cmap="YlGnBu", cbar=(layer_idx == num_layers - 1))
            
            # Fixed Title to match correct reverse layer index visually
            ax2.set_title(f"Layer {layer_idx + 1}\nLong-term Memory", fontsize=14)
            ax2.set_xlabel("Memory Slots", fontsize=12)
            
            if layer_idx == 0:
                ax2.set_ylabel("Input Sequence (0-13)", fontsize=14)
            else:
                ax2.set_ylabel("")
            ax2.tick_params(axis='y', rotation=0)

        # Adjust layout
        plt.tight_layout()
        
        # Save the image, named by Batch index (e.g., Batch00, Batch01...)
        filename = f"bf_re_Batch{batch_idx:02d}_{head_str}.png"
        save_path = os.path.join(save_dir, filename)
        
        # Save image
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # [CRITICAL!] Close the canvas to free up memory
        plt.close(fig)

    print(f"Successfully generated all {batch_size} heatmaps!")
    print(f"They are saved in directory: ./{save_dir}/")

if __name__ == "__main__":
    plot_all_batches_heatmaps()