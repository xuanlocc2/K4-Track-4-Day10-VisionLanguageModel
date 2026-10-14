"""Test phần thuần Python của lab.py: python -m pytest -q test_lab.py"""
import json

from lab import compute_reward, objects_to_answer, yolo_to_box
from scripts.score import parse_output

GT = [{"label": "pothole", "bbox": (100.0, 200.0, 300.0, 400.0)}]
PERFECT = '[{"label":"pothole","bbox_2d":[100,200,300,400]}]'


def test_reward_perfect():
    assert compute_reward(PERFECT, GT) == 1.0


def test_reward_broken_json():
    assert compute_reward('[{"label":"pothole","bbox_2d":[100,200', GT) == 0.0


def test_reward_wrong_class():
    assert compute_reward(PERFECT.replace("pothole", "crack"), GT) == 0.2


def test_reward_extra_box_penalised():
    extra = '[{"label":"pothole","bbox_2d":[100,200,300,400]},{"label":"crack","bbox_2d":[0,0,50,50]}]'
    assert abs(compute_reward(extra, GT) - 0.6) < 1e-9


def test_reward_both_empty():
    assert compute_reward("[]", []) == 1.0


def test_reward_hacks_do_not_pay():
    assert compute_reward("[]", GT) == 0.2  # trả rỗng khi có hư hại
    assert compute_reward('[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]', GT) == 0.2  # box khổng lồ


def test_label_roundtrip_and_order():
    objs = [{"label": "crack", "yolo": [0.5, 0.8, 0.2, 0.1]},
            {"label": "pothole", "yolo": [0.25, 0.3, 0.1, 0.2]}]
    ans = json.loads(objects_to_answer(objs))
    assert [a["label"] for a in ans] == ["pothole", "crack"]  # sắp theo y1
    assert ans[0]["bbox_2d"] == [200, 200, 300, 400]
    # quy ngược về pixel 640x360 phải trùng box gốc (sai số làm tròn ≤ 1 px)
    x1, y1, x2, y2 = (v / 1000 * s for v, s in zip(ans[1]["bbox_2d"], (640, 360, 640, 360)))
    assert abs(x1 - 0.4 * 640) <= 1 and abs(y2 - 0.85 * 360) <= 1
    assert parse_output(objects_to_answer(objs)) is not None and objects_to_answer([]) == "[]"


def test_box_clamped_and_non_degenerate():
    assert yolo_to_box("crack", [0.0, 1.0, 0.0001, 0.2])["bbox_2d"] == [0, 900, 1, 1000]
