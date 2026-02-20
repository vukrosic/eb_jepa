import subprocess
import time
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

epochs_to_test = [5, 10, 15, 25, 40]
results = []

print("Starting time vs accuracy experiments...")
print("Running the highly optimized config (rapid.yaml).")

for ep in epochs_to_test:
    folder_name = f"checkpoints/image_jepa/exp_{ep}ep"
    
    start = time.time()
    cmd = [
        "python", "-m", "examples.image_jepa.main", 
        "--fname", "examples/image_jepa/cfgs/rapid.yaml", 
        f"--optim.epochs={ep}", 
        f"--meta.model_folder={folder_name}",
        "--logging.tqdm_silent=True"  # Suppress tqdm
    ]
    env = os.environ.copy()
    env["EBJEPA_DSETS"] = "./datasets"
    
    print(f"Running {ep:2d} epochs... ", end="", flush=True)
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        "Epochs": ep,
        "Time (Minutes)": round(total_time / 60.0, 2),
        "Val Accuracy (%)": final_acc,
        "Train Loss": round(final_train_loss, 4) if isinstance(final_train_loss, float) else final_train_loss
    })
    print(f"Done in {total_time/60:4.1f}m -> Acc: {final_acc}% | Loss: {final_train_loss}")

print("\n--- Summary ---")
df_res = pd.DataFrame(results)
print(df_res.to_markdown(index=False))

with open("docs/fast_training_guide.md", "a") as f:
    f.write("\n\n## ⏱️ Training Time vs. Accuracy Trade-off (RTX 4090)\n\n")
    f.write("To help decide on the optimal training length for your daily research, here are the empirical results of running the **rapid config** on an RTX 4090:\n\n")
    f.write(df_res.to_markdown(index=False))
    f.write("\n\n*Note: Time includes all overhead (data loading, validation linear probing every epoch, etc).*")
