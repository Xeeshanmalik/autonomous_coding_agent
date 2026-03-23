import os
import re
import subprocess
import requests
import json

# Connect directly to your internal LLM Bridge
LLM_URL = os.getenv("LLM_BASE_URL", "http://local-qwen-backend:8082/v1") + "/chat/completions"
MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder-7b")

def run_cmd(cmd, timeout=300):
    """Executes a shell command and returns the output."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # Protect LLM VRAM by forcing scripts to use CPU
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"TimeoutError: Script execution exceeded {timeout} seconds!"

def extract_val_loss(output):
    """Parses the val_loss from the execution output."""
    # Handle 'val_loss: 0.1', 'val_loss = 0.1', 'val_loss 0.1', ignoring case
    match = re.search(r"val_loss[\s:=]+([0-9.]+)", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return float('inf')

def main():
    print("[*] Booting Local Qwen Agent Loop with Real-Time Streaming...")
    
    print("[*] Running initial baseline evaluation...")
    baseline_output = run_cmd("python train.py")
    best_loss = extract_val_loss(baseline_output)
    if best_loss == float('inf'):
        print("[-] Baseline evaluation failed or did not print val_loss. Output:")
        print(baseline_output.strip() or "No output")
    print(f"[*] Baseline val_loss established: {best_loss}")

    with open("program.md", "r") as f:
        program_instructions = f.read()

    iteration = 1
    max_iterations = int(os.environ.get("MAX_ITERATIONS", 5))
    error_feedback = ""

    while iteration <= max_iterations:
        print(f"\n" + "="*50)
        print(f"--- AutoResearch Cycle {iteration} ---")
        
        with open("train.py", "r") as f:
            current_code = f.read()

        print("[*] Querying Qwen-7B for architectural improvements...\n")
        
        user_content = f"{program_instructions}\n\nCurrent train.py:\n```python\n{current_code}\n```"
        if error_feedback:
            user_content += f"\n\nCRITICAL ERROR FEEDBACK FROM PREVIOUS RUN:\n{error_feedback}"
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an elite, autonomous quantitative researcher. Output ONLY the raw Python code inside ```python blocks. Do not explain your code. CRITICAL: YOUR SCRIPT WILL BE DISCARDED if it does not print exactly `print(f'val_loss {score}')` at the end. The AutoResearch engine ALWAYS MINIMIZES this score. You MUST intelligently select a metric that fits the user's objective (e.g., MSE for regression, `1 - F1` for classification, or `Wasserstein Distance` for synthetic data) so that LOWER is BETTER! ALWAYS read datasets dynamically using `import os; data_file = os.environ.get('DATASET_PATH', 'dataset.csv')`. CRITICAL: NEVER use hardcoded paths like 'original_data.csv' or 'synthetic_data.csv' to read/write! Use the environment variable only. ALWAYS assume the last column is the target variable and extract it using `iloc[:, -1]`; never use hardcoded column names. YOU MUST handle or drop non-numeric columns (like Dates/Strings) BEFORE numerical preprocessing like scaling or imputation. CRITICAL: You must explicitly write out `import os` and ALL required `import` declarations (like OneHotEncoder from sklearn) at the very top of your script. CRITICAL: Set `n_jobs=1` for all models/GridSearchCV. NEVER use `n_jobs=-1` as it will exhaust system RAM. NEVER use GPU parameters; the training sandbox is strictly CPU-only! CRITICAL FLAG: You may install missing modules or fix bugs based on ERROR FEEDBACK if provided. INNOVATION DIRECTIVE: You MUST drastically mutate the architecture every cycle. Do not just make tiny hyperparameter tweaks. Attempt SOTA algorithms (XGBoost, LightGBM, CatBoost, deep models), advanced feature engineering (target encoding, clustering), and robust ensemble architectures. Do not output boring linear models! Be relentlessly creative!"},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.7,
            "stream": True
        }
        
        try:
            response = requests.post(LLM_URL, json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            print("\033[96m", end="") # Cyan text
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            # HARDENED PARSING: Use .get() and explicitly check for None
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content')
                            
                            if content is not None:
                                print(content, end='', flush=True) 
                                full_response += content
                        except json.JSONDecodeError:
                            pass
                            
            print("\033[0m\n") # Reset color
            
            match = re.search(r'```(?:python)?\s*\n(.*?)```', full_response, re.DOTALL | re.IGNORECASE)
            if not match:
                print("[-] Agent failed to format code properly. Skipping cycle.")
                iteration += 1
                continue
            
            new_code = match.group(1).strip()
            with open("train.py", "w") as f:
                f.write(new_code)
                
            print("[*] Executing newly invented train.py...")
            output = run_cmd("python train.py")
            new_loss = extract_val_loss(output)
            
            if new_loss == float('inf'):
                print("[-] Agent's script failed or did not print val_loss. Output:")
                print(output.strip() or "No output")
                
                # Check for ModuleNotFoundError to auto-install
                mod_match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", output)
                if mod_match:
                    missing_mod = mod_match.group(1)
                    print(f"[*] Auto-healing: Installing missing module '{missing_mod}'...")
                    run_cmd(f"pip install {missing_mod}")
                    print(f"[*] Retrying evaluation after installing {missing_mod}...")
                    output = run_cmd("python train.py")
                    new_loss = extract_val_loss(output)
                    
                if new_loss == float('inf'):
                    print("[-] Script still failing. Saving error for self-correction in next cycle.")
                    error_feedback = f"{str(output).strip()}\nPlease FIX THIS EXACT ERROR in your next iteration."
                    # Do NOT revert code so it can fix its current broken iteration
                    iteration += 1
                    continue
            
            error_feedback = "" # reset on success
            print(f"[*] Evaluated new val_loss: {new_loss}")
            
            if new_loss < best_loss:
                print(f"[+] BREAKTHROUGH! Loss improved from {best_loss} to {new_loss}.")
                best_loss = new_loss
            else:
                print("[-] No improvement (or degraded). Reverting to previous state.")
                with open("train.py", "w") as f:
                    f.write(current_code)
                
            iteration += 1
                
        except Exception as e:
            print(f"\n[-] API or Execution Error: {e}")
            with open("train.py", "w") as f:
                f.write(current_code)
            
        iteration += 1

    print(f"\n" + "="*50)
    print(f"[*] AutoResearch Loop Completed! Final val_loss: {best_loss}")

if __name__ == "__main__":
    main()