#!/usr/bin/env python3
"""
Script chạy toàn bộ pipeline từ đầu đến cuối
Chạy tất cả các bước: EDA -> Preprocessing -> Training -> Evaluation -> Prediction
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description):
    """Chạy một script và hiển thị kết quả"""
    print("\n" + "=" * 80)
    print(f"CHẠY: {description}")
    print("=" * 80)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Không tìm thấy file: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"✅ Hoàn thành: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Đã dừng: {description}")
        return False

def main():
    """Chạy toàn bộ pipeline"""
    print("=" * 80)
    print("CHẠY TOÀN BỘ PIPELINE - DỰ ĐOÁN ĐIỆN TIÊU THỤ")
    print("=" * 80)
    
    scripts = [
        ("01_eda_analysis.py", "EDA Analysis"),
        ("02_data_preprocessing.py", "Data Preprocessing"),
        ("03_train_models.py", "Train Models"),
        ("04_evaluate_models.py", "Evaluate Models"),
        ("05_predict.py", "Prediction")
    ]
    
    results = {}
    
    for script_name, description in scripts:
        success = run_script(script_name, description)
        results[description] = success
        
        if not success:
            print(f"\n⚠️  Pipeline dừng lại ở: {description}")
            print("   Bạn có muốn tiếp tục với các bước tiếp theo? (y/n): ", end="")
            choice = input().strip().lower()
            if choice != 'y':
                break
    
    # Tóm tắt kết quả
    print("\n" + "=" * 80)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 80)
    
    for description, success in results.items():
        status = "✅ Thành công" if success else "❌ Thất bại"
        print(f"{status}: {description}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 Pipeline hoàn thành thành công!")
        print("\n📁 Kết quả:")
        print("   - EDA: analysis/")
        print("   - Processed data: output/processed_data.parquet")
        print("   - Models: output/models/")
        print("   - Visualizations: output/visualizations/")
        print("   - Predictions: output/predictions.csv")
    else:
        print("\n⚠️  Pipeline có một số bước thất bại. Vui lòng kiểm tra lại.")

if __name__ == "__main__":
    main()

