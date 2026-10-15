Họ tên: Nguyễn Văn Xuân Lộc — MSSV: 22127237
`dataset_version`: d10-21e4070301

Toàn bộ số liệu E0–E5 sinh ra từ một lần chạy trên Kaggle (GPU T4 16 GB, script `kaggle_run.py`, các stage chạy tuần tự E0→E5), đã chép vào `report/evidence/`. Các câu trả lời dưới đây dùng đúng các số đó.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — (chỉ suy luận, không đo) | — | 0.074 | 0.050–0.100 | 0.97 | 0.079 | 0.023 | 0.175 |
| E1 | LoRA LLM | 33,030,144 | 6.23 | 38.4 | 0.309 | 0.271–0.349 | 1.08 | 0.236 | 0.230 | 0.551 |
| E2 | + projector | 142,135,296 | 7.83 | 39.2 | 0.355 | 0.315–0.397 | 1.01 | 0.179 | 0.298 | 0.605 |
| E3 | + LoRA ViT | 148,426,752 | 8.00 | 47.0 | 0.436 | 0.395–0.478 | 0.94 | 0.283 | 0.376 | 0.673 |
| E4 | `train_fraction` = 0.1 (trên setup E3; E3 chính là mốc 1.0) | 148,426,752 | 7.67 | 46.5 | 0.285 | 0.250–0.321 | 1.63 | 0.179 | 0.211 | 0.567 |
| E5 (tuỳ chọn) | GRPO 60 bước, khởi tạo từ E3 | 39,321,600 | 4.76 | 31.9 | 0.454 | 0.413–0.497 | 0.77 | 0.276 | 0.399 | 0.685 |

`parse_valid_rate`: E0 0.852, E1 0.994, E2 0.997, E3 1.000, E4 0.990, E5 1.000. Cột "Phút" là thời gian huấn luyện (300 bước, batch 2 × tích luỹ 4) trên T4.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Đếm bằng `lab.count_params` trên model dựng từ config (device meta, chưa lượng tử hoá): **ViT 306.2M**, **projector 109.1M** (`visual.merger` + 3 `visual.deepstack_merger_list`), **LLM 4.022B** (đã gồm embedding; `lm_head` dùng chung trọng số với embedding nên không đếm thêm). Tổng ≈ 4.44B. Mỗi ảnh 640×360 cho 220 token ảnh.
- **E1**: chỉ LoRA r=16 trên LLM, 252 module (36 lớp × 7 ma trận), **33.0M** tham số huấn luyện. ViT và projector đóng băng.
- **E2**: E1 + huấn luyện trực tiếp toàn bộ projector (fp32): 142,135,296 − 33,030,144 = 109,105,152, khớp đúng số tham số projector đếm ở trên.
- **E3**: E2 + LoRA trên ViT, thêm 96 module (348 − 252) và 6.29M tham số, tổng **148.4M** (≈ 3.3% của 4.44B).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` = **0.852** (265/311 ảnh parse được, 46 ảnh hỏng). F1 chỉ 0.074 (TP 34, FP 303, FN 549).
1. **Lặp vô hạn rồi bị cắt cụt** — 42/46 ảnh hỏng. Model sinh một lưới box xếp nối nhau cho tới hết token nên JSON không bao giờ đóng `]`. Ví dụ thô: `[{"label": "crack", "bbox_2d": [337, 500, 400, 550]}, {"label": "crack", "bbox_2d": [337, 550, 400, 600]}, {"label": "crack", "bbox_2d": [337, 600, 400, 650]}, ...` (cắt giữa chừng).
2. **Cú pháp sai** — 4/46 ảnh, ví dụ ngoặc lệch `... [287, 478, 357, 500)}]`.

Ngoài lỗi định dạng, ở phần parse được model còn **đoán sai lớp và vị trí**: crack có 211 FP trên 299 box thật (F1 crack 0.023), và 11 ảnh trả `[]`. Với model gốc, lỗi định vị là lỗi chính; lỗi định dạng chỉ chiếm 15% ảnh.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Mình lấy đúng output E3 (F1 0.436 ở thang `norm1000`) rồi chấm lại với `coord="pixel"` (`report/e3_pixel_check.json`): **F1 = 0.000** (TP 0, FP 536, FN 583). Lý do: model xuất toạ độ trên lưới 0–1000. Nếu hiểu là pixel của ảnh 640×360 thì scorer đổi sang chuẩn hoá bằng cách chia cho 640 và 360, tức toạ độ bị phóng lên 1.56 lần theo x và 2.78 lần theo y. Box bị đẩy ra ngoài ảnh nên IoU với GT luôn dưới 0.5. Lỗi này không lộ ra qua `parse_valid_rate` (vẫn bằng 1.0), chỉ làm F1 về 0, nên phải khai đúng `coord` khi chấm.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mình kiểm tra trên một mẫu train: chuỗi dài 569 token, trong đó 220 là `<image_pad>` cùng phần còn lại của prompt; chỉ **250 token** (JSON trả lời + token kết thúc) có nhãn, **319 token** bị gán -100. Nếu tính loss trên cả chuỗi:
- Token ảnh không có nhãn thực để dự đoán (embedding đến từ ViT), ép model "dự đoán" chúng chỉ tạo gradient nhiễu.
- Prompt giống hệt nhau ở mọi mẫu nên model học thuộc prompt rất nhanh, loss trung bình tụt xuống giả tạo (319/569 = 56% token thuộc loại này) và che đi loss thật của phần JSON cần tối ưu.
- Gradient bị chia cho cả token vô ích nên phần JSON nhận tín hiệu yếu hơn.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Trọng số 4-bit NF4 được lưu dạng `uint8` đóng gói, không thể có gradient hay cập nhật trực tiếp. Quan trọng hơn, đặc trưng ViT qua lượng tử hoá có thêm nhiễu mà projector phải học trên đó, trong khi ViT là phần cần đầu vào chính xác nhất. Nên mình **chỉ lượng tử hoá LLM**, giữ ViT và projector ở fp16.

Mình đã mắc lỗi thật ở đây: lần chạy đầu truyền `llm_int8_skip_modules=["visual","lm_head"]`, nhưng trong transformers 5.19 tên module đầy đủ là `model.visual.…` nên pattern không khớp và **ViT bị lượng tử hoá âm thầm** (mọi bước vẫn chạy, `parse_valid_rate` vẫn cao). Mình phát hiện qua việc kiểm tra dtype tham số ViT, đổi thành `"model.visual"`, thêm `assert` rằng không tham số ViT nào có dtype `uint8`, rồi **chạy lại E0–E3 từ đầu** (thư mục v2). Số trong bảng là sau khi sửa.

Projector phải fp32 vì: (1) `GradScaler` từ chối unscale gradient fp16 (mình gặp lỗi dtype dạng "Attempting to unscale FP16 gradients" ở bước huấn luyện đầu); (2) với lr 2e-4 và gradient nhỏ, bản cập nhật tính theo fp16 (mantissa 10 bit) dễ bị làm tròn về 0 nên trọng số không đổi. Vì vậy tham số huấn luyện được ép `.float()` trong `apply_setup`, còn forward chạy autocast fp16.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Dữ liệu là các khung hình cắt từ video, nên khung liền kề gần như giống hệt nhau (cùng đoạn đường, cùng ổ gà, cùng ánh sáng). Chia ngẫu nhiên thì ảnh test có "bản sao gần" trong train, model chỉ cần nhớ cảnh là đúng, F1 phản ánh khả năng ghi nhớ chứ không phải tổng quát hoá. Chia theo phiên thì test là các đoạn đường chưa từng thấy. Mình **chưa chạy** thí nghiệm chia ngẫu nhiên để đo mức phình, nên đây là lập luận chứ không phải số đo.

| Split | Ảnh | Phiên | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 1422 | 13 | 1074 | 1750 | 691 |
| val | 276 | 2 | 66 | 470 | 103 |
| test | 311 | 3 | 121 | 299 | 163 |

Mình đã kiểm tra giao của các tập phiên train/val/test là rỗng.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
| | F1 | CI95 | VRAM (GB) | Phút |
|---|---|---|---|---|
| E1 | 0.309 | 0.271–0.349 | 6.23 | 38.4 |
| E2 | 0.355 | 0.315–0.397 | 7.83 | 39.2 |
| E3 | 0.436 | 0.395–0.478 | 8.00 | 47.0 |

- E1 → E2: +0.046 F1, +1.6 GB, +0.8 phút. CI95 hai bên **chồng nhau** (0.315–0.349), nên riêng bước này mình không khẳng định được là có ích.
- E2 → E3: +0.081 F1, +0.2 GB, +7.8 phút. CI cũng chồng nhau một khoảng nhỏ (0.395–0.397).
- E1 → E3: +0.127 F1, CI **không chồng** (0.349 < 0.395), nên cải thiện tổng thể là đáng tin. Chi phí thêm 1.8 GB VRAM và 8.6 phút (+22%).

Kết luận: có đáng, vì ViT và projector là phần thích nghi với miền ảnh đường phố, và chi phí tăng không lớn (VRAM vẫn thoải mái trong T4 16 GB). Nhưng mỗi setup mình chỉ chạy một seed và không làm bootstrap ghép cặp, nên từng bước nhỏ chưa được chứng minh riêng lẻ.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 rộng khoảng ±0.04 (E3: 0.395–0.478, rộng 0.083; E1: 0.271–0.349, rộng 0.078; E0: 0.050–0.100). Nghĩa là hai setup cách nhau dưới ~0.08 F1 thường không phân biệt được; cụ thể E3 vs E5 (0.436 vs 0.454) và E1 vs E2 không tách được. Hơn nữa, bootstrap chọn lại theo **ảnh** nhưng 311 ảnh chỉ đến từ **3 phiên quay** và các khung liền kề tương quan mạnh, nên số mẫu độc lập thực sự ít hơn 311 và CI thật có thể còn rộng hơn các con số trên. Lớp pothole chỉ có 121 box nên F1 theo lớp còn nhiễu hơn.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E3: manhole **0.673**, crack 0.376, **pothole 0.283 (kém nhất)**; pothole có recall 0.256 và 67 FP.
- **Kích thước**: trên tập test, box pothole nhỏ nhất, trung vị 72×61 trên thang 0–1000 và chiếm 0.42% diện tích ảnh (manhole 98×67, 0.67%; crack 139×153, 1.98%). Với box nhỏ, lệch vài pixel là IoU dưới 0.5.
- **Số mẫu**: không phải nguyên nhân chính. Pothole có 1074 box train, nhiều hơn manhole (691), mà manhole vẫn tốt hơn nhiều.
- **Ngữ nghĩa**: manhole là vật thể cứng, biên rõ, hình dạng ổn định. Pothole và crack là hư hại bề mặt có biên mơ hồ và hay nằm cạnh nhau (một mảng đường hỏng có thể được gán thành nhiều box nhỏ hoặc một box lớn), nên nhãn vốn thiếu nhất quán.

Ảnh minh hoạ (xanh = GT, đỏ = dự đoán) trong `report/figures/`: `E3_sai_0.jpg` và `E3_sai_1.jpg` cho thấy model vẽ **một box crack lớn** trùm cả vùng mà GT chia thành nhiều box pothole/crack nhỏ (TP=0, nhiều FN). `E3_dung_0.jpg` cho thấy hai manhole được khoanh đúng. Các ảnh khác: `E3_dung_1.jpg`, `E3_dung_2.jpg`, `E3_sai_2.jpg`, `loss.png`, `e5_grpo.png`.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Biến đổi: **tỉ lệ dữ liệu huấn luyện** (`train_fraction`) 1.0 → 0.1 (1422 → 142 ảnh), giữ nguyên setup E3 và 300 bước. F1 giảm từ **0.436 (CI 0.395–0.478) xuống 0.285 (CI 0.250–0.321)**, FP/ảnh tăng từ 0.94 lên 1.63. Hai CI không chồng nhau nên khác biệt là thật ở mức tập test này. Loss huấn luyện cuối của run 10% chỉ **0.021** (so với ~0.50 ở E3), tức model đã **học thuộc** 142 ảnh: 300 bước × 8 mẫu = 2400 lượt mẫu, tương đương ~17 epoch, trong khi bản đầy đủ chỉ ~1.7 epoch. Vì vậy kết luận "ít dữ liệu hơn thì kém hơn" bị lẫn với "nhiều epoch hơn nên overfit". Mình chưa tách hai yếu tố này (cần giữ số epoch cố định, hoặc thêm các mốc 0.3 và 0.5). Chỉ một seed, một lần chạy, nên mình tin vào chiều hướng chứ không tin vào độ lớn chính xác.

## Nhìn lại

- **Lỗi mà công cụ chấm không bắt được**: ViT bị lượng tử hoá 4-bit do tên module skip sai (Câu 5). `check_submission.py` và scorer đều không phát hiện vì F1 vẫn ra số hợp lý. Mình chỉ phát hiện nhờ kiểm tra dtype tham số, nên giờ có `assert`. Tương tự Câu 3: sai thang toạ độ chỉ lộ ra khi chấm lại.
- **Kết luận E5**: GRPO 60 bước cho F1 0.454 so với E3 0.436, CI chồng nhau nên **không có bằng chứng cải thiện**. FP/ảnh giảm 0.94 → 0.77 (precision 0.455 → 0.501) trong khi recall giữ nguyên. Reward trung bình 10 bước đầu (0.469) và 10 bước cuối (0.436) không tăng; monitor reward hacking bắt được 8 lần sinh box "khổng lồ", không có lần trả rỗng. 60 bước là quá ngắn để thấy xu hướng.
- **Điều mình chưa hiểu rõ**: vì sao pothole không tăng đều từ E1 (0.236) xuống E2 (0.179) rồi E3 (0.283), trong khi crack và manhole tăng đều. Có thể là nhiễu trên 121 box, nhưng mình chưa kiểm chứng. Lớp `other` luôn có F1 0 vì không có mẫu, bỏ qua được.
- **Hạn chế**: một seed mỗi setup, không đo F1 với split ngẫu nhiên, không có VRAM/thời gian cho E0. Colab bị ngắt kết nối nhiều lần nên mình chuyển sang Kaggle để chạy một mạch; notebook `lab_day10.ipynb` giữ mã nhưng không còn output, bằng chứng nằm ở `report/evidence/` và `kaggle_run.py`.
