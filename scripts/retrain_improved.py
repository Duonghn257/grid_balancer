#!/usr/bin/env python3
"""
Script để retrain model với các cải thiện:
1. Giữ thêm electricity_lag24 (7% importance) để cải thiện accuracy
2. Tune hyperparameters của XGBoost
3. Early stopping để tránh overfitting
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Chạy một script và kiểm tra kết quả"""
    print("\n" + "=" * 80)
    print(f"BƯỚC: {description}")
    print("=" * 80)
    
    if not Path(script_path).exists():
        print(f"❌ Script không tồn tại: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"✅ {description} - HOÀN TẤT")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - LỖI: {e}")
        return False

def main():
    print("=" * 80)
    print("RETRAIN MODEL VỚI CÁC CẢI THIỆN")
    print("=" * 80)
    print("\n📋 Các cải thiện:")
    print("   1. Giữ thêm electricity_lag24 (7% importance)")
    print("   2. Tune hyperparameters của XGBoost")
    print("   3. Early stopping để tránh overfitting")
    print("\n💡 Mục tiêu: Giảm RMSE từ 48.55 → <40 kWh")
    
    # Step 1: Preprocessing (đã cập nhật để giữ lag24)
    if not run_script("scripts/02_data_preprocessing.py", "Data Preprocessing (với electricity_lag24)"):
        print("\n❌ Preprocessing failed. Stopping.")
        return
    
    # Step 2: Train với improved hyperparameters
    if not run_script("scripts/improve_model_accuracy.py", "Train Model với Tuned Hyperparameters"):
        print("\n❌ Training failed. Stopping.")
        return
    
    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT RETRAIN!")
    print("=" * 80)
    print("\n📊 Kết quả:")
    print("   - Model mới đã được lưu trong: output/models/xgboost_wrapped_dice.pkl")
    print("   - Model info: output/models/model_info_dice.json")
    print("\n💡 Bước tiếp theo:")
    print("   1. Test model: python src/test_model_behavior.py")
    print("   2. Test DiCE: python src/dice_usage_example.py")
    print("   3. So sánh với model cũ để xem cải thiện")

if __name__ == "__main__":
    main()
