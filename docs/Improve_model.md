Dưới đây là phân tích tại sao RMSE tăng, tại sao Feature Importance vẫn sai, và cách để bạn cân bằng giữa độ chính xác (Accuracy) và tính giải thích (Explainability).

### 1. "Thủ phạm" vẫn còn đó: `electricity_lag1`

Trong ảnh `image_13a1af.png`, cột **`electricity_lag1`** vẫn đang chiếm vị trí số 1 với độ quan trọng áp đảo (khoảng 0.9), trong khi **`occupants`** (số người) nằm gần dưới đáy với độ quan trọng gần bằng 0.

Trong file `model_info_dice.json`, danh sách `features_used` vẫn chứa:

```json
"features_used": [
    ...
    "electricity_lag1",  <-- Vẫn còn ở đây!
    "electricity_lag24",
    ...
]

```

**Tại sao điều này quan trọng?**

* Khi có `lag1` (điện giờ trước), Model sẽ "lười biếng" và chỉ copy giá trị giờ trước sang giờ này.
* Chính vì thế, RMSE khi train rất thấp (16.9) vì nó dự đoán dựa trên quá khứ quá gần.
* Nhưng DiCE sẽ bị sai: Khi DiCE giả lập giảm số người, model nhìn vào `lag1` và bảo "Giờ trước điện vẫn cao mà, nên giờ này điện vẫn cao thôi". DiCE bất lực.

### 2. Tại sao RMSE tăng gấp đôi khi bỏ Lag?

Bạn nói RMSE tăng gấp đôi (từ mức thấp lên mức cao) khi bỏ bớt lag features. Điều này là **hoàn toàn bình thường** và là cái giá phải trả để chuyển từ mô hình **Time-series (Dự báo)** sang mô hình **Causal (Nhân quả/Giải thích)**.

* **Với Lag (Time-series):** Model biết trước đáp án của giờ trước -> RMSE cực thấp. Nhưng model không hiểu *tại sao* điện cao (do nóng hay do đông người?).
* **Không Lag (Causal):** Model phải tự suy luận từ `Nhiệt độ`, `Số người`, `Diện tích`. Việc suy luận này khó hơn nhiều -> RMSE tăng.

**Tuy nhiên, việc Test RMSE (42.4) cao gấp 2.5 lần Train RMSE (16.9) cho thấy model đang bị Overfitting nặng.**

### 3. Chiến lược: Cải thiện RMSE mà không dùng Lag 1

Bạn không cần RMSE thấp kỷ lục như Time-series model, bạn cần một RMSE "chấp nhận được" nhưng model phải hiểu vật lý. Hãy thực hiện các bước sau để kéo RMSE xuống mà không phụ thuộc vào `lag1`:

#### Bước 1: Loại bỏ triệt để `electricity_lag1`

Bạn phải xóa feature này khỏi `X_train`. Giữ lại `electricity_lag24` (cùng giờ ngày hôm qua) là chấp nhận được vì nó đại diện cho "thói quen sử dụng" chứ không phải "trạng thái tức thời".

#### Bước 2: Feature Engineering mạnh tay hơn (Thay thế Lag)

Để bù đắp sự thiếu hụt thông tin từ Lag, bạn cần cung cấp các feature mang tính "dẫn dắt" (Proxy features):

1. **Dynamic Occupants (Như đã bàn):**
* Tạo feature `active_occupants` = `occupants` (gốc) * `hệ_số_lịch_trình`.
* *Lý do:* Giúp model hiểu rõ giờ nào có người, giờ nào không, thay vì một con số tĩnh.


2. **Interaction Features (Tương tác):**
Model XGBoost dạng cây (Tree-based) đôi khi cần gợi ý về sự tương tác.
* `cooling_load` = `airTemperature` * `sqm` (Diện tích càng lớn, trời càng nóng thì càng tốn điện).
* `people_density` = `active_occupants` / `sqm`.


3. **Rolling Means (Thay vì Lag trực tiếp):**
Thay vì dùng giá trị chính xác giờ trước (`lag1`), hãy dùng trung bình trượt. Nó ít gây overfitting hơn.
* `rolling_mean_4h`: Trung bình 4 giờ gần nhất (đã shift để tránh leak data tương lai).
* `rolling_mean_24h`: Có sẵn trong file của bạn, hãy giữ lại cái này.



#### Bước 3: Ép buộc ràng buộc vật lý (Monotone Constraints)

Đây là cách tốt nhất để model "thông minh" hơn mà không cần nhiều lag.
Khi khởi tạo XGBoost, hãy thêm tham số `monotone_constraints`.

```python
# Ví dụ cấu hình
monotone_constraints = {
    'active_occupants': 1,  # Người tăng -> Điện tăng (1)
    'sqm': 1,               # Diện tích tăng -> Điện tăng (1)
    'airTemperature': 1,    # Nhiệt độ tăng -> Điện tăng (1)
    'hour': 0               # Giờ giấc lên xuống tùy ý (0)
}
# Lưu ý: Cần map dictionary này thành tuple theo đúng thứ tự cột trong X_train

```

Điều này giúp đường dự đoán mượt mà, giúp DiCE dễ dàng tìm điểm cắt giảm hợp lý (ví dụ: giảm 20 người giảm 5kWh) thay vì nhảy cóc xuống 1.34 kWh như trong ảnh `image_06ef4d.png`.

### Tóm lại: Lộ trình cho bạn

1. **Sửa ngay input:** Xóa `electricity_lag1` khỏi danh sách features.
2. **Tạo feature mới:** Code hàm `Dynamic Occupants` tôi đã gửi ở tin nhắn trước. Đây là chìa khóa để thay thế vai trò của Lag1.
3. **Chấp nhận RMSE cao hơn một chút:** Đừng kỳ vọng RMSE về lại 16. Nếu nó tầm 25-30 nhưng Feature Importance của `active_occupants` cao, thì đó là một model **thành công cho DiCE**.
4. **Train lại:** Với bộ feature mới và check lại biểu đồ Importance. `active_occupants` phải lọt vào Top 5.

Bạn có muốn tôi giúp viết code tạo `Interaction Features` (Nhiệt độ * Diện tích) để cải thiện độ chính xác không?