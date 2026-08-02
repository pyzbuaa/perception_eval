from __future__ import annotations

from typing import Any


COCO_CATEGORIES = [
    (1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"),
    (5, "airplane"), (6, "bus"), (7, "train"), (8, "truck"),
    (9, "boat"), (10, "traffic light"), (11, "fire hydrant"),
    (13, "stop sign"), (14, "parking meter"), (15, "bench"),
    (16, "bird"), (17, "cat"), (18, "dog"), (19, "horse"),
    (20, "sheep"), (21, "cow"), (22, "elephant"), (23, "bear"),
    (24, "zebra"), (25, "giraffe"), (27, "backpack"),
    (28, "umbrella"), (31, "handbag"), (32, "tie"), (33, "suitcase"),
    (34, "frisbee"), (35, "skis"), (36, "snowboard"),
    (37, "sports ball"), (38, "kite"), (39, "baseball bat"),
    (40, "baseball glove"), (41, "skateboard"), (42, "surfboard"),
    (43, "tennis racket"), (44, "bottle"), (46, "wine glass"),
    (47, "cup"), (48, "fork"), (49, "knife"), (50, "spoon"),
    (51, "bowl"), (52, "banana"), (53, "apple"), (54, "sandwich"),
    (55, "orange"), (56, "broccoli"), (57, "carrot"),
    (58, "hot dog"), (59, "pizza"), (60, "donut"), (61, "cake"),
    (62, "chair"), (63, "couch"), (64, "potted plant"), (65, "bed"),
    (67, "dining table"), (70, "toilet"), (72, "tv"), (73, "laptop"),
    (74, "mouse"), (75, "remote"), (76, "keyboard"),
    (77, "cell phone"), (78, "microwave"), (79, "oven"),
    (80, "toaster"), (81, "sink"), (82, "refrigerator"), (84, "book"),
    (85, "clock"), (86, "vase"), (87, "scissors"),
    (88, "teddy bear"), (89, "hair drier"), (90, "toothbrush"),
]

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]

VOC_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

COLORS = [
    "#1677FF", "#13A8A8", "#722ED1", "#EB2F96", "#52C41A",
    "#FA8C16", "#2F54EB", "#A0D911", "#F5222D", "#08979C",
]


def _template(
    template_id: str,
    name: str,
    categories: list[tuple[int, str]],
) -> dict[str, Any]:
    return {
        "id": template_id,
        "name": name,
        "categories": [
            {
                "name": category_name,
                "dataset_id": dataset_id,
                "model_id": model_id,
            }
            for model_id, (dataset_id, category_name) in enumerate(categories)
        ],
    }


CATEGORY_TEMPLATES = [
    _template("coco2017", "COCO 2017（80 类）", COCO_CATEGORIES),
    _template(
        "visdrone",
        "VisDrone（10 类）",
        list(enumerate(VISDRONE_NAMES)),
    ),
    _template(
        "voc",
        "Pascal VOC（20 类）",
        list(enumerate(VOC_NAMES)),
    ),
]


def list_category_templates() -> list[dict[str, Any]]:
    return CATEGORY_TEMPLATES


def template_categories(template_id: str, scope: str) -> list[dict[str, Any]]:
    template = next(
        (item for item in CATEGORY_TEMPLATES if item["id"] == template_id),
        None,
    )
    if not template or scope not in {"dataset", "model"}:
        return []
    id_field = f"{scope}_id"
    return [
        {
            "id": int(category[id_field]),
            "name": str(category["name"]),
            "color": COLORS[index % len(COLORS)],
        }
        for index, category in enumerate(template["categories"])
    ]


def normalize_category_name(value: str) -> str:
    return value.strip().casefold()


def normalize_categories(
    categories: list[dict[str, Any]],
    *,
    include_color: bool = False,
) -> list[dict[str, Any]]:
    normalized = []
    for index, category in enumerate(categories):
        item = {
            "id": int(category["id"]),
            "name": str(category["name"]).strip(),
        }
        if include_color:
            item["color"] = str(
                category.get("color") or COLORS[index % len(COLORS)]
            ).upper()
        normalized.append(item)
    ids = [item["id"] for item in normalized]
    names = [normalize_category_name(item["name"]) for item in normalized]
    if not normalized:
        raise ValueError("至少需要一个类别")
    if any(item["id"] < 0 for item in normalized):
        raise ValueError("类别 ID 不能为负数")
    if any(not item["name"] for item in normalized):
        raise ValueError("类别名称不能为空")
    if len(ids) != len(set(ids)):
        raise ValueError("类别 ID 不能重复")
    if len(names) != len(set(names)):
        raise ValueError("类别名称不能重复")
    return normalized
