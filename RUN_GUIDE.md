# 🚀 Run Guide: Mistral-Powered Adaptive Interview

Follow these steps to set up and run the **CS Interview Assistant** locally.

## 📋 Prerequisites
- **Python 3.8+**
- **Node.js & npm**
- **Mistral API Key**: Required for AI-powered features. [Get it here](https://console.mistral.ai/).

---

## 🛠️ Step 1: Backend Setup

1. **Navigate to the root directory**:
   ```powershell
   cd be
   ```

2. **Create and Activate a Virtual Environment**:
   ```powershell
   python -m venv myenv
   myenv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:
   Create a `.env` file in the root `be` folder and add:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

---

## 📂 Step 2: Data Indexing (Knowledge Base)

If you are running the project for the first time or if the technical interview section is empty, you need to build the knowledge base index.

1. **Prepare Raw Data** (optional if `kb_clean.json` already exists):
   ```powershell
   python scripts/prepare_kb.py
   ```

2. **Build Mistral-Compatible Index**:
   ```powershell
   python scripts/reindex_mistral.py
   ```
   *This command creates the `data/processed/faiss_mistral` directory with the required indices.*

---

## 🖥️ Step 3: Running the Application

You will need **two terminal windows** open simultaneously.

### Terminal A: Backend
From the root `be` directory:
```powershell
python backend/app.py
```
*Note: This will automatically initialize the database in `instance/interview_prep.db`.*

### Terminal B: Frontend
From the `be/frontend` directory:
```powershell
cd frontend
npm install  # (Only required on the first run)
npm start
```

---

## 🔄 Refreshing Data
To add new data or refresh the existing knowledge base:
1. Add JSON files to `data/raw/`.
2. Run `python scripts/prepare_kb.py`.
3. Run `python scripts/reindex_mistral.py`.

## 💡 Usage Tips
- **SignIn/SignUp**: The database starts fresh. You must **Sign Up** a new user to begin.
- **Resumes**: Upload resumes directly through the application's UI. They will be indexed automatically into the `resume_faiss` folder.
- **Debugging**: The "Code Debugging" section is adaptive and will track your performance over time.

---
🚀 *Happy Practicing!*
