# Hosting FinVerifyRAG Frontend on Vercel

This project contains a standalone React + Vite frontend in the [`finverifyrag-frontend/`](./finverifyrag-frontend) directory, equipped with full interactive demo mode (simulated 10-stage verification, radar score inspection, benchmark dashboards, calibration & drift simulation, edge-case test suite, and interactive architecture).

---

## Option 1: Deploy via Vercel Web Dashboard (Recommended)

Since your code is already in GitHub ([`GaneshS-AIML-46/FInVerifyRAG`](https://github.com/GaneshS-AIML-46/FInVerifyRAG)):

1. **Commit and Push the changes to GitHub:**
   ```bash
   git add .
   git commit -m "Fix frontend build and add Vercel SPA deployment config"
   git push origin main
   ```

2. **Open Vercel:**
   - Go to [vercel.com](https://vercel.com) and log in.
   - Click **Add New...** → **Project**.
   - Select your GitHub repository: `GaneshS-AIML-46/FInVerifyRAG`.

3. **Configure Project Settings:**
   - **Root Directory**: Click **Edit** and select `finverifyrag-frontend`.
   - **Framework Preset**: `Vite` (detected automatically).
   - **Build Command**: `npm run build` (or leave default).
   - **Output Directory**: `dist` (or leave default).
   - **Install Command**: `npm install` (or leave default).

4. **Click "Deploy"**:
   - Vercel will install dependencies, compile TypeScript, run Vite build, and provide your live URL (e.g., `https://finverifyrag-frontend.vercel.app`).

> **Note**: Even if you accidentally leave Root Directory as `./` (repo root), the root `vercel.json` will automatically direct Vercel to build and serve `finverifyrag-frontend/dist`.

---

## Option 2: Deploy directly via Vercel CLI

If you want to deploy directly from your local terminal without pushing to GitHub first:

1. Open PowerShell and navigate to the frontend folder:
   ```powershell
   cd s:\FYP\M0_M10Complete_pipeline\finverifyrag-frontend
   ```

2. Run Vercel CLI (no prior global install required):
   ```powershell
   npx vercel
   ```
   Follow the interactive prompts:
   - *Set up and deploy?* `y`
   - *Which scope?* (Select your Vercel account)
   - *Link to existing project?* `n`
   - *What’s your project’s name?* `finverifyrag-frontend`
   - *In which directory is your code located?* `./`
   - *Want to modify settings?* `n`

3. Deploy to production:
   ```powershell
   npx vercel --prod
   ```

---

## What was configured & fixed:
1. **TypeScript Build Errors Resolved**:
   - Fixed broken module import in [`src/demo/demoData.ts`](./finverifyrag-frontend/src/demo/demoData.ts).
   - Fixed Recharts Tooltip parameter types in [`AtomInspector.tsx`](./finverifyrag-frontend/src/components/AtomInspector.tsx) and [`CalibrationDrift.tsx`](./finverifyrag-frontend/src/pages/CalibrationDrift.tsx).
   - Fixed safe number extraction and type casting in [`normalize.ts`](./finverifyrag-frontend/src/lib/normalize.ts).
   - Cleaned up unused imports and variables in [`Benchmark.tsx`](./finverifyrag-frontend/src/pages/Benchmark.tsx) and [`Pipeline.tsx`](./finverifyrag-frontend/src/pages/Pipeline.tsx).
   - Hoisted impure `Date` execution out of render function in [`Benchmark.tsx`](./finverifyrag-frontend/src/pages/Benchmark.tsx) for React Compiler compatibility.
2. **SPA Routing Configuration**:
   - Added [`finverifyrag-frontend/vercel.json`](./finverifyrag-frontend/vercel.json) with client-side rewrites to prevent 404s on page reloads across all subroutes (`/pipeline`, `/benchmark`, `/calibration`, `/test-suite`, `/architecture`).
   - Added root [`vercel.json`](./vercel.json) as a fallback in case the repo is imported without setting the subfolder root in Vercel.
