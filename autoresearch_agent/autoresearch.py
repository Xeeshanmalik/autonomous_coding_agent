import os
import re
import subprocess
import requests
import json
import time

# ---------------------------------------------------------------------------
# LLM Endpoint Configuration
# ---------------------------------------------------------------------------
if os.environ.get("USE_GEMINI") == "true":
    LLM_URL = "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
    MODEL = "models/gemini-2.0-flash"
    API_KEY = os.environ.get("GEMINI_API_KEY", "")
else:
    LLM_URL = os.getenv("LLM_BASE_URL", "http://local-deepseek-backend:8080/v1") + "/chat/completions"
    MODEL = os.getenv("LLM_MODEL", "deepSeek-R1-Distill-Qwen-32B")
    API_KEY = "dummy"


# ---------------------------------------------------------------------------
# System Prompt (shared between outer research loop and self-healer)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an elite, autonomous quantitative researcher. "
    "Output ONLY the raw Python code inside ```python blocks. Do not explain your code. "
    "CRITICAL: YOUR SCRIPT WILL BE DISCARDED if it does not print exactly `print(f'val_loss {score}')` at the end. "
    "The AutoResearch engine ALWAYS MINIMIZES this score. You MUST intelligently select a metric that fits the user's objective "
    "(e.g., `mean_squared_error` from sklearn for regression, `1 - f1_score` for classification, or "
    "`scipy.stats.wasserstein_distance` for synthetic data) so that LOWER is BETTER! "
    "CRITICAL FOR METRICS: NEVER guess or hallucinate metric function imports "
    "(e.g. `jensen_shannon_divergence` does NOT exist in sklearn or scipy.stats). "
    "Use EXACTLY `scipy.stats.wasserstein_distance` if you need a distribution metric. "
    "If the provided baseline code lacks a metric, YOU MUST write one. "
    "ALWAYS read datasets dynamically using `import os; data_file = os.environ.get('DATASET_PATH', 'dataset.csv')`. "
    "CRITICAL: NEVER use hardcoded paths like 'original_data.csv' or 'synthetic_data.csv' to read/write! "
    "Use the environment variable only. "
    "ALWAYS assume the last column is the target variable and extract it using `iloc[:, -1]`; "
    "never use hardcoded column names. "
    "YOU MUST handle or drop non-numeric columns (like Dates/Strings) BEFORE numerical preprocessing like scaling or imputation. "
    "CRITICAL: You must explicitly write out `import os` and ALL required `import` declarations at the very top of your script. "
    "Ensure all code is strictly compatible with standard Python 3.9 modules (pandas, numpy, scipy, sklearn). "
    "DO NOT import or use experimental libraries like `ctgan` or `copulas` because they WILL fail. "
    "CRITICAL FOR ERROR-FREE CODE: If you receive ERROR FEEDBACK with a `TypeError` about unexpected keyword arguments, "
    "completely REMOVE the offending argument in the next iteration. DO NOT GUESS parameter names. "
    "To avoid ImportErrors, only import standard, well-known functions. "
    "CRITICAL: Set `n_jobs=1` for all models/GridSearchCV. NEVER use `n_jobs=-1` as it will exhaust system RAM. "
    "NEVER use GPU parameters; the training sandbox is strictly CPU-only! "
    "CRITICAL FLAG: You may install missing modules or fix bugs based on ERROR FEEDBACK if provided. "
    "INNOVATION DIRECTIVE: The user dataset ONLY contains numerical data. "
    "DO NOT import or use `OneHotEncoder` or `LabelEncoder` under any circumstances. "
    "Assume the data is 100% numerical out of the box. Do not write complex code to handle categorical variables. "
    "For predictive tasks, ALWAYS consider independent (X) and dependent (y) variables separately. "
    "You MUST make error-free, mathematical amendments to the provided baseline code "
    "(such as improving the data distributions, adding realistic noise correlations, or upgrading the basic RandomForest/XGBoost logic) "
    "rather than rewriting it from scratch. "
    "Maximize accuracy through robust but SIMPLE, standard library code."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd, timeout=300):
    """Execute a shell command, returning combined stdout+stderr. CPU-only sandbox."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # Protect LLM VRAM by forcing scripts to CPU
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"TimeoutError: Script execution exceeded {timeout} seconds!"


def extract_val_loss(output):
    """Parse val_loss from script output. Returns inf if not found."""
    match = re.search(r"val_loss[\s:=]+([0-9.]+)", output, re.IGNORECASE)
    return float(match.group(1)) if match else float("inf")


def extract_code_block(llm_text):
    """
    Pull the first ```python … ``` block out of LLM output.
    Falls back to everything after </think> for reasoning models.
    Returns the code string, or None if nothing usable is found.
    """
    # Strip <think>…</think> reasoning before searching
    text = re.sub(r"<think>.*?</think>", "", llm_text, flags=re.DOTALL)

    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: raw text after </think>
    if "</think>" in llm_text:
        tail = llm_text.split("</think>")[-1].strip()
        if tail:
            return tail

    return None


def query_llm(messages, stream=True):
    """
    Send a chat-completion request to the configured LLM endpoint.

    Handles:
      - Streaming responses (prints tokens in cyan, accumulates full text).
      - Exponential-backoff retry on HTTP 429.

    Returns the full accumulated response text, or raises on unrecoverable error.
    """
    headers = {"Content-Type": "application/json"}
    if os.environ.get("USE_GEMINI") == "true":
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "stream": stream,
    }

    retry_count, max_retries, backoff = 0, 3, 10
    while True:
        response = requests.post(LLM_URL, json=payload, headers=headers, stream=stream)
        if response.status_code == 429 and retry_count < max_retries:
            print(f"\n[-] Rate limit hit (429). Retrying in {backoff}s… ({retry_count + 1}/{max_retries})")
            time.sleep(backoff)
            retry_count += 1
            backoff *= 2
            continue
        response.raise_for_status()
        break

    full_response = ""
    print("\033[96m", end="")  # Cyan text for LLM output

    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content is not None:
                        print(content, end="", flush=True)
                        full_response += content
                except json.JSONDecodeError:
                    pass

    print("\033[0m\n")  # Reset colour

    # Persist raw LLM response for offline debugging
    with open("last_llm_response.log", "w") as f:
        f.write(full_response)

    return full_response


# ---------------------------------------------------------------------------
# Self-Healing Inner Loop
# ---------------------------------------------------------------------------

def execute_and_heal(initial_code, max_retries=3):
    """
    Run `initial_code` as train.py and attempt to self-heal on failure.

    Algorithm
    ---------
    1. Write `code` to train.py and execute it via subprocess.
    2. If the process exits cleanly (returncode == 0) → return (stdout, code).
    3. If the process fails:
       a. Capture the full stderr traceback.
       b. Build a targeted repair prompt containing the broken code + traceback.
       c. Query the LLM for a corrected script.
       d. Replace `code` with the LLM's fix and go back to step 1.
    4. After `max_retries` failed healing attempts, log the failure and return None.

    Parameters
    ----------
    initial_code : str   – The Python source code to execute.
    max_retries  : int   – Maximum LLM-assisted fix attempts before giving up.

    Returns
    -------
    (stdout_output : str, final_code : str)  on success.
    None                                      if all healing attempts are exhausted.
    """
    code = initial_code
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # Keep training strictly on CPU

    for attempt in range(max_retries + 1):  # attempt 0 = first run, 1..max = heals

        # --- Write the current code to disk ---
        with open("train.py", "w") as f:
            f.write(code)

        label = "initial execution" if attempt == 0 else f"heal attempt {attempt}/{max_retries}"
        print(f"[*] Running train.py ({label})…")

        # --- Execute: capture stdout and stderr SEPARATELY for precise diagnosis ---
        try:
            proc = subprocess.run(
                ["python", "train.py"],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print("[-] Script timed out after 300 s.")
            if attempt == max_retries:
                print("[-] Self-healer exhausted all retries (timeout). Giving up.")
                return None
            # Feed timeout as error context so LLM can optimise runtime
            stderr_snapshot = "TimeoutError: Script exceeded 300-second execution limit. Optimise for speed."
        else:
            stdout_snapshot = proc.stdout
            stderr_snapshot = proc.stderr

            if proc.returncode == 0:
                # ✅ Success — return the captured stdout and the working code
                print("[+] Script executed successfully.")
                return stdout_snapshot, code

            # Non-zero exit: print the traceback for the operator log
            print(f"[-] Script exited with code {proc.returncode}.")
            if stderr_snapshot.strip():
                print("    STDERR ↓")
                print(stderr_snapshot.strip())

        # --- Healing exhausted? ---
        if attempt == max_retries:
            print(f"[-] Self-healer exhausted all {max_retries} repair attempts. Giving up on this code.")
            return None

        # --- Build a surgical repair prompt for the LLM ---
        print(f"[*] Querying LLM for a surgical fix (heal {attempt + 1}/{max_retries})…")

        repair_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "The following Python script raised a runtime error. "
                    "Your ONLY task is to return the FULLY CORRECTED script inside a ```python block. "
                    "Do NOT explain anything. Do NOT change the algorithm logic — fix ONLY the bug.\n\n"
                    f"=== BROKEN CODE ===\n```python\n{code}\n```\n\n"
                    f"=== FULL ERROR TRACEBACK ===\n{stderr_snapshot}\n\n"
                    "Return the complete, corrected Python script now."
                ),
            },
        ]

        try:
            llm_response = query_llm(repair_messages)
        except Exception as e:
            print(f"[-] LLM query failed during healing: {e}")
            return None

        fixed_code = extract_code_block(llm_response)
        if not fixed_code:
            print("[-] LLM did not return a parseable code block. Retrying healing prompt…")
            # Keep `code` unchanged so we retry with the same broken version + same error
            continue

        print("[*] LLM returned a candidate fix. Applying and re-testing…")
        code = fixed_code  # Use the fixed code in the next loop iteration


# ---------------------------------------------------------------------------
# Outer Evolutionary Research Loop
# ---------------------------------------------------------------------------

def main():
    print("[*] Booting AutoResearch Agent with Self-Healing Inner Loop…")

    # --- Baseline evaluation ---
    print("[*] Running initial baseline evaluation…")
    baseline_output = run_cmd("python train.py")
    best_loss = extract_val_loss(baseline_output)
    if best_loss == float("inf"):
        print(
            "[!] Baseline script did not output a 'val_loss'. "
            "The AI will invent an appropriate metric and beat Infinity."
        )
    print(f"[*] Baseline val_loss established: {best_loss}")

    with open("program.md", "r") as f:
        program_instructions = f.read()

    iteration = 1
    max_iterations = int(os.environ.get("MAX_ITERATIONS", 5))
    best_code_snapshot = open("train.py").read()  # Keep a copy of the last-known-good code

    while iteration <= max_iterations:
        print(f"\n{'='*50}")
        print(f"--- AutoResearch Cycle {iteration} ---")

        # Read the current champion code
        with open("train.py", "r") as f:
            current_code = f.read()

        # Rate-limit mitigation for Gemini
        if iteration > 1 and os.environ.get("USE_GEMINI") == "true":
            print("[*] Rate-limit mitigation: sleeping 5 s before next cycle…")
            time.sleep(5)

        model_name = "Gemini-2.0-Flash" if os.environ.get("USE_GEMINI") == "true" else "DeepSeek-32B"
        print(f"[*] Querying {model_name} for architectural improvements…\n")

        # --- Build the outer research prompt ---
        user_content = (
            f"{program_instructions}\n\nCurrent train.py:\n```python\n{current_code}\n```"
        )

        research_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # --- Query the LLM for an improved candidate ---
        try:
            llm_response = query_llm(research_messages)
        except Exception as e:
            print(f"\n[-] LLM query failed: {e}")
            # Restore champion code so the file is never left in a broken state
            with open("train.py", "w") as f:
                f.write(current_code)
            iteration += 1
            continue

        new_code = extract_code_block(llm_response)
        if not new_code:
            print("[-] Agent failed to format code properly. Skipping cycle.")
            iteration += 1
            continue

        # -----------------------------------------------------------------------
        # 🔧 SELF-HEALING INNER LOOP
        # Pass the LLM's new code to execute_and_heal(). It will run it, and if
        # it crashes, feed the traceback back to the LLM for up to 3 repair
        # attempts before declaring the cycle a failure.
        # -----------------------------------------------------------------------
        heal_result = execute_and_heal(new_code, max_retries=3)

        if heal_result is None:
            # All healing attempts failed — revert to the last known-good code
            print("[-] Self-healer could not recover the script. Reverting to previous champion.")
            with open("train.py", "w") as f:
                f.write(current_code)
            iteration += 1
            continue

        exec_output, healed_code = heal_result
        new_loss = extract_val_loss(exec_output)

        if new_loss == float("inf"):
            # Script ran without error but forgot to print val_loss
            print("[-] Script completed but did not emit 'val_loss'. Reverting.")
            print(exec_output.strip() or "No output")
            with open("train.py", "w") as f:
                f.write(current_code)
            iteration += 1
            continue

        print(f"[*] Evaluated new val_loss: {new_loss}")

        if new_loss < best_loss:
            print(f"[+] BREAKTHROUGH! Loss improved: {best_loss:.6f} → {new_loss:.6f}")
            best_loss = new_loss
            # Write the winner (healed_code may differ from new_code if healing ran)
            with open("train.py", "w") as f:
                f.write(healed_code)
        else:
            print(f"[-] No improvement ({new_loss:.6f} ≥ {best_loss:.6f}). Reverting to previous champion.")
            with open("train.py", "w") as f:
                f.write(current_code)

        iteration += 1

    print(f"\n{'='*50}")
    print(f"[*] AutoResearch Loop Completed! Final val_loss: {best_loss}")


if __name__ == "__main__":
    main()