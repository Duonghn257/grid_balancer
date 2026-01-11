# Giải thích Dataset: Building Data Genome Project 2

## 🚀 Tóm tắt Nhanh

**Dataset này dùng để làm gì?**
- Dự đoán lượng điện tiêu thụ của các tòa nhà dựa trên đặc điểm của tòa nhà và thời tiết

**3 file chính:**
1. **metadata.csv**: Thông tin tòa nhà (diện tích, số người, loại sử dụng, v.v.) - **1,636 buildings**
2. **electricity_cleaned.csv**: Lượng điện tiêu thụ theo giờ - **~17,544 giờ × 1,578 buildings**
3. **weather.csv**: Dữ liệu thời tiết theo giờ - **Nhiệt độ, mây, gió, mưa, v.v.**

**Features quan trọng nhất:**
- ⭐⭐⭐ `sqm` (diện tích), `occupants` (số người), `primaryspaceusage` (loại sử dụng), `airTemperature` (nhiệt độ)
- ⭐⭐ `yearbuilt` (năm xây), `numberoffloors` (số tầng), `timezone` (múi giờ), `hour` (giờ trong ngày)

---

## 📋 Tổng quan về Dataset

**Building Data Genome Project 2 (BDG2)** là một dataset mở về dữ liệu năng lượng từ các tòa nhà phi dân cư. Dataset bao gồm:

- **3,053 energy meters** từ **1,636 buildings**
- **Thời gian**: 2 năm đầy đủ (2016 và 2017)
- **Tần suất**: Đo theo giờ (hourly measurements)
- **Các loại meters**: Điện, nước nóng, nước lạnh, hơi nước, nước, tưới tiêu, năng lượng mặt trời, gas
- **19 sites** ở Bắc Mỹ và Châu Âu

---

## 📁 Cấu trúc Dataset

Dataset bao gồm 3 file chính:

1. **`metadata.csv`**: Thông tin tĩnh về các buildings
2. **`electricity_cleaned.csv`**: Dữ liệu time-series về lượng điện tiêu thụ
3. **`weather.csv`**: Dữ liệu thời tiết theo giờ

---

## 🏢 METADATA.CSV - Thông tin về Buildings

File này chứa **thông tin tĩnh** về mỗi building (không thay đổi theo thời gian). Mỗi dòng đại diện cho một building.

### Cấu trúc: 1,636 buildings × 31 features

### 📊 Chi tiết các Features:

#### **1. Định danh Building**
- **`building_id`** (string): Mã định danh duy nhất của building
  - Format: `SiteID_PrimaryUsage_UniqueName`
  - Ví dụ: `Panther_lodging_Dean`, `Fox_education_Maria`
  - **Ý nghĩa**: Dùng để join với các file khác

- **`site_id`** (string): Mã định danh của site (khu vực)
  - Ví dụ: `Panther`, `Fox`, `Robin`, `Rat`, `Bear`, `Lamb`, `Peacock`, `Moose`, `Gator`, `Bull`, `Bobcat`, `Crow`, `Shrew`, `Swan`, `Wolf`, `Hog`, `Eagle`, `Cockatoo`
  - **Ý nghĩa**: Các buildings trong cùng site thường có đặc điểm tương tự (vị trí địa lý, khí hậu)

- **`building_id_kaggle`** (float): ID số cho Kaggle competition (có thể null)
- **`site_id_kaggle`** (float): ID số của site cho Kaggle competition (có thể null)

#### **2. Thông tin Địa lý**
- **`lat`** (float): Vĩ độ của building (latitude)
  - **Ý nghĩa**: Ảnh hưởng đến khí hậu, thời tiết → ảnh hưởng đến nhu cầu năng lượng

- **`lng`** (float): Kinh độ của building (longitude)
  - **Ý nghĩa**: Tương tự như latitude

- **`timezone`** (string): Múi giờ của site
  - Ví dụ: `US/Eastern`, `US/Central`, `US/Pacific`, `Europe/London`
  - **Ý nghĩa**: Ảnh hưởng đến pattern sử dụng năng lượng (giờ cao điểm khác nhau)

#### **3. Loại Sử dụng Building**
- **`primaryspaceusage`** (string): Loại sử dụng chính của building
  - Các giá trị: `Education`, `Office`, `Lodging/residential`, `Assembly`, `Public`, `Retail`, `Parking`, `Warehouse`, `Food`, `Health`, `Science`, `Industrial`, `Services`, `Other`, `Unknown`
  - **Ý nghĩa**: ⭐ **RẤT QUAN TRỌNG** - Mỗi loại building có pattern tiêu thụ năng lượng khác nhau
    - Education: Tiêu thụ cao vào giờ học
    - Office: Tiêu thụ cao vào giờ làm việc
    - Lodging: Tiêu thụ ổn định 24/7

- **`sub_primaryspaceusage`** (string): Phân loại chi tiết hơn
  - Ví dụ: `Classroom`, `Research`, `Residence Hall`, `Office`, `Retail Store`
  - **Ý nghĩa**: Chi tiết hóa loại sử dụng, có thể ảnh hưởng đến pattern tiêu thụ

#### **4. Đặc điểm Vật lý Building**
- **`sqm`** (float): Diện tích sàn của building (square meters)
  - **Ý nghĩa**: ⭐ **RẤT QUAN TRỌNG** - Diện tích lớn hơn → nhu cầu năng lượng cao hơn
  - Đơn vị: m²

- **`sqft`** (float): Diện tích sàn của building (square feet)
  - Tương đương với `sqm`, chỉ khác đơn vị

- **`numberoffloors`** (float): Số tầng của building
  - **Ý nghĩa**: ⭐ **QUAN TRỌNG** - Số tầng nhiều hơn → cần nhiều năng lượng hơn (thang máy, HVAC, v.v.)

- **`yearbuilt`** (float): Năm xây dựng building
  - **Ý nghĩa**: ⭐ **QUAN TRỌNG** - Buildings cũ hơn thường kém hiệu quả năng lượng hơn
  - Format: YYYY (ví dụ: 1989, 2008, 2016)

- **`date_opened`** (string): Ngày mở cửa sử dụng
  - Format: D/M/YYYY
  - **Ý nghĩa**: Có thể khác với năm xây dựng

#### **5. Số lượng Người sử dụng**
- **`occupants`** (float): Số người thường xuyên sử dụng building
  - **Ý nghĩa**: ⭐ **RẤT QUAN TRỌNG** - Nhiều người hơn → nhu cầu năng lượng cao hơn (điều hòa, đèn, thiết bị, v.v.)

#### **6. Loại Meters có trong Building**
Các cột này cho biết building có loại meter nào (Yes/NaN):
- **`electricity`** (string): Có meter điện hay không
  - **Ý nghĩa**: Chỉ các buildings có `electricity = "Yes"` mới có dữ liệu trong `electricity_cleaned.csv`
  - **1,578 buildings** có electricity meter

- **`hotwater`**: Meter nước nóng
- **`chilledwater`**: Meter nước lạnh (điều hòa)
- **`steam`**: Meter hơi nước
- **`water`**: Meter nước
- **`irrigation`**: Meter tưới tiêu
- **`solar`**: Meter năng lượng mặt trời
- **`gas`**: Meter gas

#### **7. Thông tin Ngành nghề**
- **`industry`** (string): Loại ngành nghề
  - Ví dụ: `Education`, `Healthcare`, `Commercial`, `Industrial`
  - **Ý nghĩa**: Có thể ảnh hưởng đến pattern sử dụng năng lượng

- **`subindustry`** (string): Phân loại chi tiết hơn của ngành nghề

#### **8. Hệ thống Sưởi**
- **`heatingtype`** (string): Loại hệ thống sưởi
  - **Ý nghĩa**: Ảnh hưởng đến tiêu thụ năng lượng vào mùa đông

#### **9. Đánh giá Hiệu quả Năng lượng**
- **`energystarscore`** (string): Điểm Energy Star (1-100)
  - **Ý nghĩa**: ⭐ **QUAN TRỌNG** - Điểm cao hơn = hiệu quả năng lượng tốt hơn
  - Chỉ có 163 buildings có điểm này

- **`eui`** (string): Energy Use Intensity (kWh/năm/m²)
  - **Ý nghĩa**: ⭐ **RẤT QUAN TRỌNG** - Chỉ số tiêu thụ năng lượng chuẩn hóa theo diện tích
  - Đây là một chỉ số quan trọng để đánh giá hiệu quả năng lượng

- **`site_eui`** (string): Energy Use Intensity của toàn site (kWh/năm/m²)
- **`source_eui`** (string): Primary energy consumption chuẩn hóa (kWh/năm/m²)

- **`leed_level`** (string): LEED rating (Leadership in Energy and Environmental Design)
  - Ví dụ: `Certified`, `Silver`, `Gold`, `Platinum`
  - **Ý nghĩa**: Buildings có LEED rating thường hiệu quả năng lượng hơn

- **`rating`** (string): Các đánh giá năng lượng khác

---

## ⚡ ELECTRICITY_CLEANED.CSV - Dữ liệu Tiêu thụ Điện

File này chứa **dữ liệu time-series** về lượng điện tiêu thụ theo giờ cho mỗi building.

### Cấu trúc:
- **Cột đầu tiên**: `timestamp` (datetime) - Thời gian đo
- **Các cột còn lại**: Mỗi cột là một `building_id` (tên building)
- **Giá trị**: Lượng điện tiêu thụ tại thời điểm đó (kWh)
- **Số dòng**: ~17,544 dòng (2 năm × 365 ngày × 24 giờ)

### Format: Wide Format (mỗi building là một cột)

```
timestamp,Panther_parking_Lorriane,Panther_lodging_Cora,Panther_office_Hannah,...
2016-01-01 00:00:00,0.0,0.0,0.0,...
2016-01-01 01:00:00,26.96,15.72,70.75,...
2016-01-01 02:00:00,0.0,16.08,74.31,...
```

### Ý nghĩa:
- **`timestamp`**: Thời điểm đo (theo giờ)
- **Giá trị trong mỗi cột**: Lượng điện tiêu thụ (kWh) tại thời điểm đó
- **NaN/trống**: Building không có dữ liệu tại thời điểm đó

### Cách sử dụng:
1. **Chuyển sang Long Format**: Melt để có cấu trúc `[timestamp, building_id, electricity_consumption]`
2. **Tính toán thống kê**: 
   - Trung bình theo building: `groupby('building_id')['electricity_consumption'].mean()`
   - Tổng theo building: `groupby('building_id')['electricity_consumption'].sum()`
   - Pattern theo giờ/ngày/tuần/tháng

---

## 🌤️ WEATHER.CSV - Dữ liệu Thời tiết

File này chứa **dữ liệu thời tiết theo giờ** cho mỗi site.

### Cấu trúc:
- **Cột đầu tiên**: `timestamp` (datetime) - Thời gian đo
- **Cột thứ hai**: `site_id` (string) - Mã định danh site
- **Các cột còn lại**: Các thông số thời tiết

### 📊 Các Features Thời tiết:

#### **1. Nhiệt độ**
- **`airTemperature`** (float): Nhiệt độ không khí (°C)
  - **Ý nghĩa**: ⭐ **RẤT QUAN TRỌNG** - Nhiệt độ cao → cần nhiều điều hòa → tiêu thụ điện cao
  - Nhiệt độ thấp → cần sưởi ấm → tiêu thụ điện cao

- **`dewTemperature`** (float): Nhiệt độ điểm sương (°C)
  - **Ý nghĩa**: Ảnh hưởng đến độ ẩm, có thể ảnh hưởng đến cảm giác nhiệt độ

#### **2. Áp suất**
- **`seaLvlPressure`** (float): Áp suất mực nước biển (hPa hoặc mbar)
  - **Ý nghĩa**: Có thể ảnh hưởng đến hiệu suất HVAC

#### **3. Gió**
- **`windSpeed`** (float): Tốc độ gió (m/s hoặc km/h)
  - **Ý nghĩa**: Gió mạnh có thể giúp làm mát tự nhiên → giảm nhu cầu điều hòa

- **`windDirection`** (float): Hướng gió (độ)
  - **Ý nghĩa**: Ít quan trọng hơn windSpeed

#### **4. Mây và Mưa**
- **`cloudCoverage`** (float): Độ che phủ mây (%)
  - **Ý nghĩa**: Mây nhiều → ít ánh nắng → ít nóng → giảm nhu cầu điều hòa

- **`precipDepth1HR`** (float): Lượng mưa trong 1 giờ (mm)
  - **Ý nghĩa**: Mưa có thể làm mát → giảm nhu cầu điều hòa

- **`precipDepth6HR`** (float): Lượng mưa trong 6 giờ (mm)

### Ý nghĩa tổng thể:
Thời tiết **ảnh hưởng rất lớn** đến tiêu thụ năng lượng:
- **Mùa hè nóng**: Nhu cầu điều hòa cao → tiêu thụ điện cao
- **Mùa đông lạnh**: Nhu cầu sưởi ấm cao → tiêu thụ điện cao
- **Mùa xuân/thu**: Nhiệt độ ôn hòa → tiêu thụ điện thấp hơn

---

## 🎯 Features Quan trọng cho Bài toán Dự đoán Năng lượng Điện

### ⭐⭐⭐ RẤT QUAN TRỌNG (Must-have):

1. **`sqm`** (diện tích): Diện tích lớn hơn → nhu cầu năng lượng cao hơn
2. **`occupants`** (số người): Nhiều người hơn → nhu cầu năng lượng cao hơn
3. **`primaryspaceusage`** (loại sử dụng): Mỗi loại có pattern khác nhau
4. **`airTemperature`** (nhiệt độ): Ảnh hưởng trực tiếp đến nhu cầu điều hòa/sưởi
5. **`eui`** (nếu có): Chỉ số hiệu quả năng lượng chuẩn hóa

### ⭐⭐ QUAN TRỌNG (Should-have):

6. **`yearbuilt`**: Buildings cũ thường kém hiệu quả hơn
7. **`numberoffloors`**: Số tầng nhiều → nhu cầu cao hơn
8. **`site_id`**: Các buildings trong cùng site có đặc điểm tương tự
9. **`timezone`**: Ảnh hưởng đến pattern sử dụng theo giờ
10. **`sub_primaryspaceusage`**: Chi tiết hóa loại sử dụng
11. **`cloudCoverage`**, **`windSpeed`**: Ảnh hưởng đến nhu cầu điều hòa

### ⭐ CÓ THỂ HỮU ÍCH (Nice-to-have):

12. **`energystarscore`**: Điểm hiệu quả năng lượng (nhưng chỉ có 163 buildings)
13. **`leed_level`**: LEED rating
14. **`heatingtype`**: Loại hệ thống sưởi
15. **`dewTemperature`**: Nhiệt độ điểm sương
16. **`precipDepth1HR`**: Lượng mưa

### 📅 Features Thời gian (có thể tạo từ timestamp):

- **`hour`**: Giờ trong ngày (0-23) - ⭐⭐ QUAN TRỌNG
- **`day_of_week`**: Ngày trong tuần (0-6) - ⭐⭐ QUAN TRỌNG
- **`month`**: Tháng (1-12) - ⭐⭐ QUAN TRỌNG
- **`is_weekend`**: Cuối tuần hay không - ⭐ QUAN TRỌNG
- **`season`**: Mùa (Spring, Summer, Fall, Winter) - ⭐⭐ QUAN TRỌNG

---

## 💡 Gợi ý cho Bài toán Dự đoán Năng lượng Điện

### 1. **Features nên sử dụng cho Model:**

```python
# Continuous features (số)
continuous_features = [
    'sqm',                    # Diện tích
    'yearbuilt',              # Năm xây dựng
    'numberoffloors',         # Số tầng
    'occupants',              # Số người
    'airTemperature',         # Nhiệt độ (từ weather)
    'cloudCoverage',          # Độ che phủ mây
    'windSpeed',              # Tốc độ gió
    'hour',                   # Giờ trong ngày (tạo từ timestamp)
    'day_of_week',            # Ngày trong tuần
    'month'                   # Tháng
]

# Categorical features (danh mục)
categorical_features = [
    'primaryspaceusage',      # Loại sử dụng chính
    'sub_primaryspaceusage',  # Phân loại chi tiết
    'site_id',                # Site
    'timezone',               # Múi giờ
    'season'                  # Mùa (tạo từ month)
]
```

### 2. **Target Variable:**

- **Cho bài toán dự đoán tổng thể**: Sử dụng `avg_electricity` (trung bình lượng điện tiêu thụ của building)
- **Cho bài toán dự đoán theo thời gian**: Sử dụng `electricity_consumption` tại từng thời điểm

### 3. **Lưu ý về Missing Values:**

- **`occupants`**: Chỉ có 230/1636 buildings có dữ liệu (14%)
- **`yearbuilt`**: Chỉ có 817/1636 buildings có dữ liệu (50%)
- **`numberoffloors`**: Chỉ có 441/1636 buildings có dữ liệu (27%)
- **`eui`**: Chỉ có 299/1636 buildings có dữ liệu (18%)

→ Cần xử lý missing values cẩn thận (imputation, hoặc chỉ sử dụng features có đủ dữ liệu)

### 4. **Kết hợp dữ liệu:**

```python
# 1. Load electricity data và tính trung bình cho mỗi building
building_electricity = df_electricity.groupby('building_id')['electricity_consumption'].mean()

# 2. Merge với metadata
df = pd.merge(building_electricity, metadata, on='building_id')

# 3. Merge với weather data (theo site_id và timestamp)
# Có thể tính trung bình weather theo site hoặc theo thời gian
```

---

## 📚 Tài liệu Tham khảo

- [Building Data Genome Project 2 - GitHub](https://github.com/buds-lab/building-data-genome-project-2)
- [Energy Star Building Types](https://www.energystar.gov/buildings/facility-owners-and-managers/existing-buildings/use-portfolio-manager/identify-your-property-type)
- [ASHRAE Great Energy Predictor III Competition](https://www.kaggle.com/c/ashrae-energy-prediction)

---

## 🔍 Checklist khi Làm việc với Dataset

- [ ] Kiểm tra missing values trong mỗi feature
- [ ] Xác định features nào có đủ dữ liệu để sử dụng
- [ ] Tạo features thời gian từ timestamp (hour, day_of_week, month, season)
- [ ] Merge electricity data với metadata
- [ ] Merge weather data với building data (theo site_id)
- [ ] Xử lý outliers trong electricity consumption
- [ ] Chuẩn hóa/normalize continuous features
- [ ] Encode categorical features (one-hot encoding hoặc label encoding)

---

**Tác giả**: Tài liệu này được tạo để hỗ trợ bài toán dự đoán năng lượng điện sử dụng DiCE (Diverse Counterfactual Explanations)

