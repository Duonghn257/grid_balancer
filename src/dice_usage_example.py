#!/usr/bin/env python3
"""
Ví dụ sử dụng DiCE Explainer đúng cách cho use case:
- Dự đoán lượng điện tiêu thụ trong tương lai
- Nếu vượt threshold → Gợi ý điều chỉnh vừa đủ để không quá tải
"""

import json
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from dice_explainer import DiceExplainer

def main():
    print("=" * 80)
    print("VÍ DỤ: DỰ ĐOÁN VÀ GỢI Ý ĐIỀU CHỈNH ĐỂ TRÁNH QUÁ TẢI")
    print("=" * 80)
    
    # Khởi tạo DiCE Explainer
    explainer = DiceExplainer()
    
    # ========================================================================
    # BƯỚC 1: DỮ LIỆU TÒA NHÀ VÀ THỜI TIẾT
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 1: DỮ LIỆU TÒA NHÀ")
    print("=" * 80)
    
    building_data = {
        'time': '2016-01-01T21:00:00',  # Thời điểm muốn dự đoán
        'building_id': 'Bear_education_Sharon',
        'site_id': 'Bear',
        'primaryspaceusage': 'Education',
        'sub_primaryspaceusage': 'Education',
        'sqm': 5261.7,
        'yearbuilt': 1953,
        'numberoffloors': 5,
        'occupants': 200,  # Số người sử dụng
        'timezone': 'US/Pacific',
        # Weather data
        'airTemperature': 25.0,
        'cloudCoverage': 30.0,
        'dewTemperature': 18.0,
        'windSpeed': 2.6,
        'seaLvlPressure': 1020.7,
        'precipDepth1HR': 0.0
    }
    
    print(f"📋 Thông tin tòa nhà:")
    print(f"   • Building ID: {building_data['building_id']}")
    print(f"   • Diện tích: {building_data['sqm']:.1f} m²")
    print(f"   • Số người: {building_data['occupants']}")
    print(f"   • Thời điểm: {building_data['time']}")
    
    # ========================================================================
    # BƯỚC 2: DỰ ĐOÁN LƯỢNG ĐIỆN TIÊU THỤ
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 2: DỰ ĐOÁN LƯỢNG ĐIỆN TIÊU THỤ")
    print("=" * 80)
    
    prediction = explainer.inference.predict(building_data, include_lag=True)
    print(f"\n📊 Dự đoán tiêu thụ: {prediction:.2f} kWh")
    
    # ========================================================================
    # BƯỚC 3: KIỂM TRA NGUỠNG THRESHOLD
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 3: KIỂM TRA NGUỠNG THRESHOLD")
    print("=" * 80)
    
    # Ngưỡng tối đa cho phép (ví dụ: công suất lưới điện)
    THRESHOLD = 50.0  # kWh
    print(f"\n🎯 Ngưỡng tối đa cho phép: {THRESHOLD} kWh")
    
    if prediction <= THRESHOLD:
        print(f"\n✅ AN TOÀN - Không vượt ngưỡng")
        print(f"   Dự đoán ({prediction:.2f} kWh) < Ngưỡng ({THRESHOLD} kWh)")
        return
    
    # Vượt ngưỡng
    excess = prediction - THRESHOLD
    reduction_needed = (excess / prediction) * 100
    print(f"\n⚠️ VƯỢT NGUỠNG!")
    print(f"   • Dự đoán: {prediction:.2f} kWh")
    print(f"   • Ngưỡng: {THRESHOLD} kWh")
    print(f"   • Vượt: {excess:.2f} kWh ({reduction_needed:.1f}%)")
    print(f"   • Cần giảm: {excess:.2f} kWh để an toàn")
    
    # ========================================================================
    # BƯỚC 4: TẠO GỢI Ý ĐIỀU CHỈNH
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 4: TẠO GỢI Ý ĐIỀU CHỈNH")
    print("=" * 80)
    
    print(f"\n🔍 Đang tạo gợi ý điều chỉnh...")
    result = explainer.generate_recommendations(
        json_data=building_data,
        threshold=THRESHOLD,
        total_cfs=5,
        method='random'  # Nhanh hơn 'genetic'
    )
    
    if not result['success']:
        print(f"\n❌ Lỗi: {result.get('error', 'Unknown error')}")
        return
    
    # ========================================================================
    # BƯỚC 5: LỌC VÀ HIỂN THỊ GỢI Ý THỰC TẾ
    # ========================================================================
    print("\n" + "=" * 80)
    print("BƯỚC 5: GỢI Ý ĐIỀU CHỈNH THỰC TẾ")
    print("=" * 80)
    
    # Lọc recommendations thực tế (gần threshold, không quá cực đoan)
    # Chỉ lấy những cái trong khoảng 90-100% của threshold
    realistic_min = THRESHOLD * 0.9
    realistic_recs = [
        rec for rec in result['recommendations']
        if rec['predicted_consumption'] >= realistic_min
        and rec['predicted_consumption'] <= THRESHOLD
    ]
    
    # Sắp xếp theo độ gần threshold (gần nhất trước)
    realistic_recs.sort(key=lambda r: abs(r['predicted_consumption'] - THRESHOLD))
    
    if realistic_recs:
        print(f"\n✅ Tìm thấy {len(realistic_recs)} gợi ý thực tế (giảm vừa đủ):")
        print(f"   (Chỉ hiển thị các gợi ý trong khoảng {realistic_min:.1f}-{THRESHOLD} kWh)")
        
        for i, rec in enumerate(realistic_recs[:3], 1):
            print(f"\n   {'─' * 70}")
            print(f"   💡 Gợi ý {i}:")
            print(f"      • Tiêu thụ sau điều chỉnh: {rec['predicted_consumption']:.2f} kWh")
            print(f"      • Giảm: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
            print(f"      • Trạng thái: {'✅ Dưới ngưỡng' if rec['below_threshold'] else '❌ Vẫn vượt'}")
            
            # Chỉ hiển thị các features thực sự có thể điều chỉnh
            actionable_changes = [
                ch for ch in rec.get('changes', [])
                if ch['feature'] in ['occupants']  # Chỉ occupants là thực tế
            ]
            
            if actionable_changes:
                print(f"      • Cần điều chỉnh:")
                for change in actionable_changes:
                    print(f"        - {change['action']}")
                    print(f"          ({change['description']})")
            else:
                print(f"      • (Không có thay đổi features có thể điều chỉnh thực tế)")
    else:
        print(f"\n⚠️ Không tìm thấy gợi ý thực tế gần threshold")
        print(f"   DiCE chỉ tìm được các gợi ý cực đoan (giảm quá nhiều)")
        print(f"\n   Tất cả recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"   {i}. Giảm xuống {rec['predicted_consumption']:.2f} kWh ({rec['reduction_pct']:.1f}% reduction)")
            print(f"      (Quá cực đoan - không thực tế)")
    
    # ========================================================================
    # TÓM TẮT
    # ========================================================================
    print("\n" + "=" * 80)
    print("TÓM TẮT")
    print("=" * 80)
    print(f"\n📊 Dự đoán ban đầu: {prediction:.2f} kWh")
    print(f"🎯 Ngưỡng tối đa: {THRESHOLD} kWh")
    print(f"📉 Cần giảm: {excess:.2f} kWh ({reduction_needed:.1f}%)")
    
    if realistic_recs:
        best_rec = realistic_recs[0]
        print(f"\n✅ Gợi ý tốt nhất:")
        print(f"   • Giảm xuống: {best_rec['predicted_consumption']:.2f} kWh")
        print(f"   • Giảm: {best_rec['reduction']:.2f} kWh ({best_rec['reduction_pct']:.1f}%)")
    else:
        print(f"\n⚠️ Không có gợi ý thực tế - cần điều chỉnh threshold hoặc features")

if __name__ == "__main__":
    main()
