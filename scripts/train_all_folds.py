#!/usr/bin/env python3
"""
Train model on all 5 folds sequentially.
Each fold's best model is saved to outputs/fold_X/model_best.keras

Usage:
    poetry run python scripts/train_all_folds.py
"""
import os
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    n_folds = 5
    base_output_dir = Path("outputs")
    base_output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("5-FOLD CROSS-VALIDATION TRAINING")
    print("=" * 60)
    
    for fold in range(n_folds):
        fold_dir = base_output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold + 1}/{n_folds}")
        print(f"Output directory: {fold_dir}")
        print(f"{'='*60}\n")
        
        # Run training with fold override
        cmd = [
            "poetry", "run", "python", "-m", "src.train",
            "experiment=production",
            f"train.fold={fold}",
            f"hydra.run.dir={fold_dir}"
        ]
        
        result = subprocess.run(cmd, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"❌ Fold {fold} failed with return code {result.returncode}")
            sys.exit(1)
        
        # Copy best model to fold directory
        best_model = Path("model_best.keras")
        if best_model.exists():
            dest = fold_dir / "model_best.keras"
            shutil.copy(best_model, dest)
            print(f"✅ Saved best model to {dest}")
        
        # Copy OOF predictions
        oof_file = Path("oof.csv")
        if oof_file.exists():
            dest = fold_dir / "oof.csv"
            shutil.copy(oof_file, dest)
            print(f"✅ Saved OOF predictions to {dest}")
    
    print("\n" + "=" * 60)
    print("✅ ALL FOLDS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: make ensemble")
    print("  2. Run: make submit-final")


if __name__ == "__main__":
    main()
