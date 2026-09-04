#!/usr/bin/env python3
"""
AffectSense AI - Protected Client Distribution Builder
======================================================
Compiles and obfuscates the core engine (detector, sincerity classifier,
biometrics, and cryptographic auth manager) using PyArmor into encrypted
binary bytecode to protect against source code tampering and piracy.

Usage:
  python build_protected.py           # Build protected folder in dist/
  python build_protected.py --zip     # Build and package into a distributable .zip
"""

import os
import sys
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DIST_DIR = ROOT_DIR / "dist" / "AffectSense-AI-Protected"
ZIP_OUTPUT = ROOT_DIR / "dist" / "AffectSense-AI-Protected.zip"

def build():
    parser = argparse.ArgumentParser(description="Build protected distribution of AffectSense AI")
    parser.add_argument("--zip", action="store_true", help="Create a distributable .zip archive after building")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  🛡️  AFFECTSENSE AI - PROTECTED CLIENT BUILDER")
    print("=" * 65)

    # 1. Check PyArmor
    pyarmor_bin = ROOT_DIR / "venv" / "bin" / "pyarmor"
    if not pyarmor_bin.exists():
        pyarmor_bin = Path(shutil.which("pyarmor") or "pyarmor")
    
    print("\n[Step 1/5] Checking PyArmor compiler...")
    res = subprocess.run([str(pyarmor_bin), "--version"], capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR] PyArmor is not installed or available. Run 'pip install pyarmor' first.")
        sys.exit(1)
    print("  ✓ PyArmor ready.")

    # 2. Clean previous build
    print("\n[Step 2/5] Preparing output directory...")
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output target: {DIST_DIR}")

    # 3. Obfuscate Core Engine
    print("\n[Step 3/5] Obfuscating core engine and classification algorithms...")
    cmd = [
        str(pyarmor_bin), "gen",
        "-O", str(DIST_DIR),
        "-r", "engine"
    ]
    gen_res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    if gen_res.returncode != 0:
        print(f"[ERROR] PyArmor build failed:\n{gen_res.stderr or gen_res.stdout}")
        sys.exit(1)
    print("  ✓ Core sensory & classification engine encrypted into tamper-proof binary.")

    # 4. Copy Application & Assets
    print("\n[Step 4/5] Copying application runners and required assets...")
    
    # Runners
    shutil.copy2(ROOT_DIR / "app.py", DIST_DIR / "app.py")
    if (ROOT_DIR / "live_sense.py").exists():
        shutil.copy2(ROOT_DIR / "live_sense.py", DIST_DIR / "live_sense.py")
    
    # Requirements
    if (ROOT_DIR / "requirements.txt").exists():
        shutil.copy2(ROOT_DIR / "requirements.txt", DIST_DIR / "requirements.txt")
    
    # .env.example and default client .env (strictly locked by default)
    if (ROOT_DIR / ".env.example").exists():
        shutil.copy2(ROOT_DIR / ".env.example", DIST_DIR / ".env.example")
        # Create default client .env
        client_env = (
            "# AffectSense AI - Client Deployment Configuration\n"
            "AUTH_ENABLED=true\n"
            "SESSION_EXPIRATION_MINUTES=10\n"
            "ALLOW_PUBLIC_KEY_GENERATION=false\n"
            "KEY_GENERATION_COOLDOWN_SECONDS=60\n"
            "ADMIN_MASTER_KEY=affectsense_admin_secure_key\n"
            "AUTH_SIGNING_SECRET=AffectSense-AI-Core-Signing-Seed-v1-998234-SECURE\n"
        )
        (DIST_DIR / ".env").write_text(client_env)

    # Models
    dist_models = DIST_DIR / "models"
    dist_models.mkdir(exist_ok=True)
    src_model = ROOT_DIR / "models" / "face_landmarker.task"
    if src_model.exists():
        shutil.copy2(src_model, dist_models / "face_landmarker.task")
        print("  ✓ MediaPipe model bundled.")
    
    # Sample data
    src_samples = ROOT_DIR / "sample_data"
    if src_samples.exists():
        shutil.copytree(src_samples, DIST_DIR / "sample_data", dirs_exist_ok=True)
        print("  ✓ Sample benchmark media bundled.")

    # Client README
    client_readme = (
        "# AffectSense AI - Client Studio Edition\n\n"
        "## Setup & Launch Instructions\n\n"
        "1. **Install Dependencies**:\n"
        "   ```bash\n"
        "   python -m venv venv\n"
        "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate\n"
        "   pip install -r requirements.txt\n"
        "   ```\n\n"
        "2. **Launch Studio**:\n"
        "   ```bash\n"
        "   streamlit run app.py\n"
        "   ```\n\n"
        "3. **Access Key**:\n"
        "   This package requires an authorized Access Key provided by the distributor.\n"
        "   Enter your key on the verification screen to begin your session.\n"
    )
    (DIST_DIR / "README.md").write_text(client_readme)
    print("  ✓ Client documentation bundled.")

    # 5. Optional Zip Packaging
    if args.zip:
        print("\n[Step 5/5] Compressing into distributable .zip archive...")
        if ZIP_OUTPUT.exists():
            ZIP_OUTPUT.unlink()
        
        with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(DIST_DIR):
                for f in files:
                    full_path = Path(root) / f
                    rel_path = full_path.relative_to(DIST_DIR.parent)
                    zipf.write(full_path, rel_path)
        
        size_mb = ZIP_OUTPUT.stat().st_size / (1024 * 1024)
        print(f"  ✓ Distributable package created: {ZIP_OUTPUT} ({size_mb:.2f} MB)")
    else:
        print("\n[Step 5/5] Skipping .zip (pass --zip to package archive).")

    print("\n" + "=" * 65)
    print("  🎉 BUILD COMPLETE! Output ready at:")
    print(f"     {DIST_DIR}")
    if args.zip:
        print(f"     {ZIP_OUTPUT}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    build()
