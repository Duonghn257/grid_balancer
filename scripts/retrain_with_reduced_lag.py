#!/usr/bin/env python3
"""
Script retrain XGBoost với ít lag features hơn
- Chỉ giữ lại electricity_lag1 (87% importance)
- Bỏ các lag features khác để model học mối quan hệ với features khác tốt hơn
- Chỉ lấy data từ 2017-10-01 trở đi
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a script and handle errors"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_path}")
        print(f"Return code: {result.returncode}")
        return False
    
    return True

def main():
    print("="*80)
    print("RETRAIN XGBOOST VỚI REDUCED LAG FEATURES")
    print("="*80)
    print("\n📋 Thay đổi:")
    print("   1. Chỉ giữ lại electricity_lag1 (87% importance)")
    print("   2. Bỏ các lag features khác")
    print("   3. Chỉ lấy data từ 2017-10-01 trở đi")
    print("\n⚠️  LƯU Ý: Script này sẽ:")
    print("   - Ghi đè output/processed_data.parquet")
    print("   - Ghi đè output/models/xgboost_wrapped_dice.pkl")
    print("   - Ghi đè output/models/label_encoders_dice.pkl")
    print("   - Ghi đè output/models/model_info_dice.json")
    print("   - Ghi đè output/features_info.json")
    
    response = input("\n❓ Bạn có muốn tiếp tục? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Đã hủy")
        return
    
    scripts_dir = Path("scripts")
    
    # Step 1: Preprocess data (với filter date và reduced lag features)
    if not run_script(scripts_dir / "02_data_preprocessing.py", "BƯỚC 1: PREPROCESS DATA"):
        print("\n❌ Preprocessing failed!")
        return
    
    # Step 2: Train XGBoost model
    if not run_script(scripts_dir / "06_train_xgboost_for_dice.py", "BƯỚC 2: TRAIN XGBOOST MODEL"):
        print("\n❌ Training failed!")
        return
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT RETRAIN!")
    print("="*80)
    print("\n📋 Kết quả:")
    print("   - Model mới đã được lưu vào: output/models/xgboost_wrapped_dice.pkl")
    print("   - Features info đã được cập nhật: output/features_info.json")
    print("\n🔍 Tiếp theo:")
    print("   1. Test model mới: python src/test_model_behavior.py")
    print("   2. Test DiCE: python src/dice_usage_example.py")
    print("   3. So sánh feature importance - occupants nên có importance cao hơn")

if __name__ == "__main__":
    main()
