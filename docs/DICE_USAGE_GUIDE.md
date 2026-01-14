# Hướng dẫn sử dụng DiCE Explainer

## 📖 Giải thích các Features

### `is_weekend` là gì?

`is_weekend` là một **binary feature** (0 hoặc 1) cho biết thời điểm dự đoán có phải là cuối tuần không:

- **`is_weekend = 1`**: Là cuối tuần (Thứ 7 hoặc Chủ nhật)
- **`is_weekend = 0`**: Là ngày trong tuần (Thứ 2 - Thứ 6)

**Công thức tính:**
```python
is_weekend = 1 if day_of_week >= 5 else 0
# day_of_week: 0=Monday, 1=Tuesday, ..., 5=Saturday, 6=Sunday
```

**Tại sao quan trọng?**
- Cuối tuần thường có pattern tiêu thụ điện khác với ngày trong tuần
- Ví dụ: Tòa nhà văn phòng tiêu thụ ít điện hơn vào cuối tuần (ít người làm việc)

**⚠️ Lưu ý:** 
- `is_weekend` là feature **không thể điều chỉnh thực tế** - bạn không thể "biến" thứ 3 thành cuối tuần
- DiCE đang đề xuất thay đổi feature này là **không hợp lý** cho use case thực tế

---

## 🎯 Use Case của bạn

### Mục tiêu:
1. **Dự đoán** lượng điện tiêu thụ trong tương lai
2. Nếu **vượt ngưỡng threshold** → Gợi ý điều chỉnh để **không quá tải**
3. Điều chỉnh **vừa đủ** để dưới threshold, không cần giảm tối đa

### Vấn đề hiện tại:

DiCE đang tối ưu để đạt **giá trị thấp nhất** trong range `[0, threshold]`, dẫn đến:
- ✅ Đề xuất giảm 98% (từ 87.87 kWh → 1.34 kWh)
- ❌ Không phù hợp với use case thực tế
- ❌ Người dùng chỉ cần giảm vừa đủ (từ 87.87 kWh → ~70 kWh)

---

## 💡 Cách sử dụng đúng

### 1. Dự đoán tương lai

```python
from src.dice_explainer import DiceExplainer

# Khởi tạo
explainer = DiceExplainer()

# Dữ liệu tòa nhà và thời tiết
json_data = {
    'time': '2016-01-01T21:00:00',  # Thời điểm muốn dự đoán
    'building_id': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'occupants': 200,
    'airTemperature': 25.0,
    # ... các features khác
}

# Dự đoán
current_pred = explainer.inference.predict(json_data)
print(f"Dự đoán: {current_pred:.2f} kWh")
```

### 2. Kiểm tra threshold và gợi ý điều chỉnh

```python
# Ngưỡng tối đa cho phép (ví dụ: công suất lưới điện)
THRESHOLD = 100.0  # kWh

if current_pred > THRESHOLD:
    print(f"⚠️ Vượt ngưỡng! Cần giảm {current_pred - THRESHOLD:.2f} kWh")
    
    # Tạo gợi ý điều chỉnh
    result = explainer.generate_recommendations(
        json_data=json_data,
        threshold=THRESHOLD,
        total_cfs=5,
        method='random'  # Nhanh hơn 'genetic'
    )
    
    if result['success']:
        # Lọc các recommendations thực tế (gần threshold)
        realistic_recs = [
            rec for rec in result['recommendations']
            if rec['predicted_consumption'] >= THRESHOLD * 0.9  # 90-100% của threshold
        ]
        
        if realistic_recs:
            print("\n💡 Gợi ý điều chỉnh (vừa đủ để dưới threshold):")
            for rec in realistic_recs[:3]:
                print(f"\n  • Giảm xuống: {rec['predicted_consumption']:.2f} kWh")
                for change in rec['changes']:
                    print(f"    - {change['action']}")
else:
    print("✅ An toàn, không vượt ngưỡng")
```

---

## 🔧 Các Features có thể điều chỉnh

### ✅ Có thể điều chỉnh thực tế:

1. **`occupants`** (Số người)
   - **Cách điều chỉnh:** Giảm số người sử dụng tòa nhà
   - **Ví dụ:** Từ 200 → 150 người

2. **`hour`** (Giờ trong ngày) - ⚠️ Cần cẩn thận
   - **Cách điều chỉnh:** Thay đổi lịch hoạt động
   - **Ví dụ:** Chuyển hoạt động từ giờ cao điểm (21h) sang giờ thấp điểm (6h)
   - **Lưu ý:** Không thể thay đổi thời gian thực tế, chỉ có thể điều chỉnh lịch

### ❌ Không thể điều chỉnh (nhưng DiCE đang đề xuất):

1. **`is_weekend`** - Không thể "biến" ngày trong tuần thành cuối tuần
2. **`sqm`** - Không thể giảm diện tích tòa nhà
3. **`airTemperature`** - Đây là nhiệt độ môi trường (weather), không thể điều khiển

---

## 🚨 Vấn đề và Giải pháp

### Vấn đề 1: DiCE đề xuất giảm quá nhiều

**Nguyên nhân:** DiCE tối ưu để đạt giá trị thấp nhất trong range `[0, threshold]`

**Giải pháp:** Lọc recommendations để chỉ lấy những cái gần threshold:

```python
# Sau khi generate recommendations
realistic_recs = [
    rec for rec in result['recommendations']
    if rec['predicted_consumption'] >= threshold * 0.9  # 90-100% của threshold
    and rec['predicted_consumption'] <= threshold
]

# Ưu tiên những cái gần threshold nhất
realistic_recs.sort(key=lambda r: abs(r['predicted_consumption'] - threshold))
```

### Vấn đề 2: DiCE đề xuất thay đổi features không thể điều chỉnh

**Nguyên nhân:** Một số features (như `is_weekend`, `sqm`) đang được đánh dấu là có thể điều chỉnh

**Giải pháp:** Đã được sửa trong code - các features này đã được đánh dấu là `adjustable: False`

---

## 📝 Ví dụ hoàn chỉnh

```python
#!/usr/bin/env python3
"""
Ví dụ: Dự đoán và gợi ý điều chỉnh để tránh quá tải
"""

from src.dice_explainer import DiceExplainer

# Khởi tạo
explainer = DiceExplainer()

# Dữ liệu tòa nhà
building_data = {
    'time': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sub_primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'yearbuilt': 1953,
    'numberoffloors': 5,
    'occupants': 200,
    'timezone': 'US/Pacific',
    'airTemperature': 25.0,
    'cloudCoverage': 30.0,
    'dewTemperature': 18.0,
    'windSpeed': 2.6,
    'seaLvlPressure': 1020.7,
    'precipDepth1HR': 0.0
}

# Bước 1: Dự đoán
prediction = explainer.inference.predict(building_data)
print(f"📊 Dự đoán tiêu thụ: {prediction:.2f} kWh")

# Bước 2: Kiểm tra ngưỡng
THRESHOLD = 100.0  # Ngưỡng tối đa cho phép
print(f"🎯 Ngưỡng tối đa: {THRESHOLD} kWh")

if prediction > THRESHOLD:
    excess = prediction - THRESHOLD
    print(f"⚠️ Vượt ngưỡng {excess:.2f} kWh - Cần điều chỉnh!")
    
    # Bước 3: Tạo gợi ý
    result = explainer.generate_recommendations(
        json_data=building_data,
        threshold=THRESHOLD,
        total_cfs=5,
        method='random'
    )
    
    if result['success']:
        # Bước 4: Lọc recommendations thực tế
        realistic = [
            rec for rec in result['recommendations']
            if rec['predicted_consumption'] >= THRESHOLD * 0.9
            and rec['predicted_consumption'] <= THRESHOLD
        ]
        
        if realistic:
            print(f"\n💡 Tìm thấy {len(realistic)} gợi ý thực tế:")
            for i, rec in enumerate(realistic[:3], 1):
                print(f"\n  Gợi ý {i}:")
                print(f"    • Tiêu thụ sau điều chỉnh: {rec['predicted_consumption']:.2f} kWh")
                print(f"    • Giảm: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
                
                if rec['changes']:
                    print(f"    • Cần điều chỉnh:")
                    for change in rec['changes']:
                        # Chỉ hiển thị các features thực sự có thể điều chỉnh
                        if change['feature'] in ['occupants']:  # Chỉ occupants là thực tế
                            print(f"      - {change['action']}")
                else:
                    print(f"    • (Không có thay đổi features có thể điều chỉnh)")
        else:
            print("\n⚠️ Không tìm thấy gợi ý thực tế gần threshold")
            print("   DiCE chỉ tìm được các gợi ý cực đoan (giảm quá nhiều)")
else:
    print("✅ An toàn - Không vượt ngưỡng")
```

---

## 🎓 Tóm tắt

1. **`is_weekend`**: Feature binary cho biết cuối tuần (0/1), **không thể điều chỉnh thực tế**

2. **Use case đúng:**
   - Dự đoán → Kiểm tra threshold → Nếu vượt → Gợi ý điều chỉnh **vừa đủ**

3. **Features có thể điều chỉnh:**
   - ✅ `occupants` (số người)
   - ⚠️ `hour` (giờ hoạt động - cần cẩn thận)

4. **Cách xử lý recommendations:**
   - Lọc để chỉ lấy những cái gần threshold (90-100%)
   - Bỏ qua các recommendations cực đoan (giảm >50%)
