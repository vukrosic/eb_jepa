import subprocess
import time
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

lrs_to_test = [0.1, 0.3, 0.5, 0.707, 1.0, 1.5]
epochs = 5
results = []

print(f"Starting learning rate search for {epochs} epochs...")
print("Running the highly optimized config (rapid.yaml).")

for lr in lrs_to_test:
    folder_name = f"checkpoints/image_jepa/exp_lr_{lr}"
    
    start = time.time()
    cmd = [
        "python", "-m", "examples.image_jepa.main", 
        "--fname", "examples/image_jepa/cfgs/rapid.yaml", 
        f"--optim.epochs={epochs}", 
        f"--optim.lr={lr}",
        f"--meta.model_folder={folder_name}",
        "--logging.tqdm_silent=True"  # Suppress tqdm
    ]
    env = os.environ.copy()
    env["EBJEPA_DSETS"] = "./datasets"
    
    print(f"\nRunning LR {lr}...")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"Error running for {lr}")
    end = time.time()
    
    total_time = end - start
    
    # Read metrics
    csv_path = f"{folder_name}/metrics.csv"
    try:
        df = pd.read_csv(csv_path)
        final_acc = df.iloc[-1]['val_acc']
        final_train_loss = df.iloc[-1]['train_loss']
    except Exception as e:
        final_acc = "Error"
        final_train_loss = "Error"
    
    results.append({
        "LR": lr,
        "Time (Minutes)": round(total_time / 60.0, 2),
        "Val Accuracy (%)": final_acc,
        "Train Loss": round(final_train_loss, 4) if isinstance(final_train_loss, float) else final_train_loss
    })
    print(f"Done in {total_time/60:4.1f}m -> Acc: {final_acc}% | Loss: {final_train_loss}")

print("\n--- Summary ---")
df_res = pd.DataFrame(results)
print(df_res.to_markdown(index=False))

with open("docs/fast_training_guide.md", "a") as f:
    f.write(f"\n\n## ⏱️ Learning Rate Search ({epochs} Epochs)\n\n")
    f.write("Results of the learning rate search on rapid config:\n\n")
    f.write(df_res.to_markdown(index=False))
