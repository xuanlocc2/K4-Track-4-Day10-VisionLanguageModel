# Báo cáo lỗi và lưu ý

## Lỗi đã gặp và cách xử lý
| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| ViT bị lượng tử hoá 4-bit âm thầm | transformers 5.19 khớp tên module đầy đủ, skip phải là `"model.visual"` chứ không phải `"visual"` | Sửa tên, thêm `assert` không có tham số ViT dtype `uint8`, chạy lại E0–E3 |
| `GradScaler` báo không unscale được gradient fp16 | Tham số huấn luyện đang ở fp16 | Ép tham số huấn luyện (LoRA, projector) sang fp32 |
| `ImportError` khi gắn LoRA lên ViT | peft từ chối torchao 0.10 trên Colab | Gỡ torchao |
| CUDA OOM | Tiến trình cũ còn giữ GPU | Mỗi stage một tiến trình, kiểm tra GPU trước khi chạy, `expandable_segments` |
| Colab ngắt kết nối giữa chừng | Giới hạn phiên | Checkpoint mỗi 50 bước có resume, runner nền; cuối cùng chạy một mạch trên Kaggle T4 |

## Lưu ý khi đọc kết quả
- Mỗi setup chỉ một seed; CI95 chồng nhau giữa E1/E2, E2/E3, E3/E5 nên chưa kết luận được từng bước nhỏ.
- Bootstrap theo ảnh nhưng test chỉ có 3 phiên quay, CI thực có thể rộng hơn.
- E4 lẫn hai yếu tố: ít dữ liệu và nhiều epoch hơn (300 bước cố định).
- E0 không đo VRAM và thời gian (chỉ suy luận).
- Lớp `other` trong `per_class` luôn bằng 0 vì không có mẫu.
- Notebook không còn output; bằng chứng nằm ở `report/evidence/` và `kaggle_run.py`.
