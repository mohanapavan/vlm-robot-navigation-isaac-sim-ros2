from vlm_nav.detector import Detection, box_iou, filter_detections

SHAPE = (480, 640, 3)


def det(label, score, box):
    return Detection(label, score, box)


def test_iou_basics():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(box_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_overlapping_boxes_on_one_object_are_one_sighting():
    boxes = [det('bed', 0.5, (100, 100, 300, 250)), det('bed', 0.7, (105, 98, 305, 252)),
             det('bed', 0.4, (110, 110, 200, 200))]            # third is mostly inside the first two
    (kept,) = filter_detections(boxes, SHAPE)
    assert kept.score == 0.7                                  # the best box survives


def test_one_object_cannot_be_two_classes():
    kept = filter_detections([det('desk', 0.45, (10, 10, 110, 110)), det('vending machine', 0.6, (12, 10, 112, 112))], SHAPE)
    assert [d.label for d in kept] == ['vending machine']


def test_neighbours_and_separate_objects_survive():
    a = det('chair', 0.5, (0, 0, 100, 100))
    b = det('chair', 0.5, (110, 0, 210, 100))                 # side by side, no overlap
    c = det('desk', 0.5, (90, 0, 190, 100))                   # overlaps a and b only a little
    assert len(filter_detections([a, b, c], SHAPE)) == 3


def test_image_filling_boxes_and_slivers_are_dropped():
    assert filter_detections([det('wall', 0.9, (0, 0, 640, 480))], SHAPE) == []
    assert filter_detections([det('door', 0.9, (10, 10, 15, 200))], SHAPE) == []
    assert len(filter_detections([det('door', 0.9, (10, 10, 100, 300))], SHAPE)) == 1


def test_empty_input():
    assert filter_detections([], SHAPE) == []
