import os
import re
import subprocess
import requests
import json

# Connect directly to your internal LLM Bridge
LLM_URL = os.getenv("LLM_BASE_URL", "http://local-qwen-backend:8080/v1") + "/chat/completions"
MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder-7b")

def run_cmd(cmd):
    """Executes a shell command and returns the output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def extract_val_loss(output):
    """Parses the val_loss from the execution output."""
    match = re.search(r"val_loss\s+([0-9.]+)", output)
    if match:
        return float(match.group(1))
    return float('inf')

def main():
    print("[*] Booting Local Qwen Agent Loop with Real-Time Streaming...")
    
    print("[*] Running initial baseline evaluation...")
    baseline_output = run_cmd("python train.py")
    best_loss = extract_val_loss(baseline_output)
    print(f"[*] Baseline val_loss established: {best_loss}")

    with open("program.md", "r") as f:
        program_instructions = f.read()

    iteration = 1
    while True:
        print(f"\n" + "="*50)
        print(f"--- AutoResearch Cycle {iteration} ---")
        
        with open("train.py", "r") as f:
            current_code = f.read()

        print("[*] Querying Qwen-7B for architectural improvements...\n")
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an elite, autonomous quantitative researcher. Output ONLY the raw Python code inside ```python blocks. Do not explain your code."},
                {"role": "user", "content": f"{program_instructions}\n\nCurrent train.py:\n```python\n{current_code}\n```"}
            ],
            "temperature": 0.2,
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
            
            match = re.search(r'```python\n(.*?)```', full_response, re.DOTALL)
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
            print(f"[*] Evaluated new val_loss: {new_loss}")
            
            if new_loss < best_loss:
                print(f"[+] BREAKTHROUGH! Loss improved from {best_loss} to {new_loss}.")
                run_cmd(f'git add train.py && git commit -m "Auto-improvement: val_loss {new_loss}"')
                best_loss = new_loss
            else:
                print("[-] No improvement (or script crashed). Reverting to previous state.")
                run_cmd('git reset --hard HEAD')
                
        except Exception as e:
            print(f"\n[-] API or Execution Error: {e}")
            run_cmd('git reset --hard HEAD')
            
        iteration += 1

if __name__ == "__main__":
    main()