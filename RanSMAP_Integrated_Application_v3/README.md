

\# PS-RanSMAP — Integrated Application



\## Project

\*\*Adaptive RanSMAP: An Incremental Learning Framework for Ransomware Detection Using Storage and Memory Access Patterns\*\*



This package contains the integrated RanSMAP frontend and FastAPI backend.



\## Project Structure



\- `backend/` — FastAPI application, routes, preprocessing, model services and model assets.

\- `frontend/` — HTML, CSS and JavaScript interface.


\- `requirements.txt` — Python dependencies.



\## Prerequisites



\- Python with compatibility for the required PyTorch version.

\- VS Code.

\- VS Code Live Server extension.

\- Internet access for installing dependencies.



\## Setup Instructions



\### 1. Extract the ZIP



Extract `RanSMAP\_Integrated\_Application\_v2.zip`.



Open the extracted `02\_integrated\_application` folder in VS Code.



\### 2. Create a virtual environment



Open the terminal in the project folder:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Ensure the installed PyTorch version is compatible with the included model artifacts.



\### 4. Start the backend



From the project root:



```powershell

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

```



Backend:

http://127.0.0.1:8000



API documentation:

http://127.0.0.1:8000/docs



Health endpoint:

http://127.0.0.1:8000/api/health



\### 5. Start the frontend



\- Open `frontend/index.html` in VS Code.

\- Right-click and select \*\*Open with Live Server\*\*.

\- Keep Live Server running on port 5500.



The frontend communicates with the backend running on port 8000.



\## Included Model Assets



The package includes the model and preprocessing assets currently used by the integrated backend.



These include:

\- Feature normalization statistics.

\- Baseline model artifact.

\- Threshold configuration and model artifact.



\## Current Implementation Status



\- Frontend and backend integration completed.

\- Application startup confirmed locally.

\- Adaptive model initialization confirmed locally.

\- Baseline and adaptive prediction workflows are implemented in the application.

\- Model comparison interface is implemented.



\## Known Limitations



\- Actual incremental model weight updates are not yet enabled.

\- Feedback is currently accepted in memory and is not persistent across server restarts.

\- Final adaptive training and model artifacts are pending.

\- Full end-to-end testing and experimental validation are pending.



Do not interpret successful application startup as proof of completed model validation.



\## Notes



\- The original RanSMAP dataset is not included in this package.

\- The Python virtual environment is intentionally excluded.

\- Install dependencies separately using `requirements.txt`.

\- Use the final model artifacts supplied by the model-training team when available.



