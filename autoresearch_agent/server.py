import os
import subprocess
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

@app.post("/run")
async def run_evolution(
    task: str = Form(...),
    baseline: str = Form(...),
    iterations: int = Form(...),
    modelChoice: str = Form("local"),
    apiKey: str = Form("AIzaSyDG8nPBt0pxZQeoDGZ-k1MSmaNBAbi_aVg"),
    data: UploadFile = File(None)
):
    # 1. Save the task definition
    with open("program.md", "w") as f:
        f.write(task)
        
    # 2. Save the baseline script
    with open("train.py", "w") as f:
        f.write(baseline)
        
    # 3. Save the dataset if the user uploaded one (e.g., Walmart_Sales.csv)
    if data:
        with open(data.filename, "wb") as f:
            f.write(await data.read())

    # 4. Generator to stream the researcher's output back to React
    def stream_researcher():
        # Pass the UI's iteration slider value to the script
        env = os.environ.copy()
        env["MAX_ITERATIONS"] = str(iterations)
        if modelChoice == "gemini":
            env["USE_GEMINI"] = "true"
            env["GEMINI_API_KEY"] = apiKey
        if data:
            env["DATASET_PATH"] = data.filename
        
        process = subprocess.Popen(
            ["python", "-u", "autoresearch.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env
        )
        
        while True:
            read_fn = getattr(process.stdout, "read1", process.stdout.read)
            chunk = read_fn(128)
            if not chunk:
                break
            yield chunk.decode('utf-8', errors='replace')
            
        process.stdout.close()
        process.wait()

        # Transmit the final train.py state exactly as text demarcated by special tags
        try:
            with open("train.py", "r") as f:
                yield "\n[FINAL_CODE_START]\n"
                for code_line in f:
                    yield code_line
                yield "\n[FINAL_CODE_END]\n"
        except FileNotFoundError:
            pass

    return StreamingResponse(stream_researcher(), media_type="text/plain")

if __name__ == "__main__":
    # Listen on all interfaces so the Mac host can reach it
    uvicorn.run(app, host="0.0.0.0", port=8000)