import enum


class DirectValueMeta(enum.EnumMeta):
    """https://muellerzr.github.io/blog/HackingTheEnum.html"""

    def __getattribute__(cls, name):
        value = super().__getattribute__(name)
        if isinstance(value, cls):
            value = value.value

        return value


class TaskEnum(enum.StrEnum, metaclass=DirectValueMeta):
    input = "input"
    depth = "depth"
    road_detection = "road_detection"
    object_detection_2d = "object_detection_2d"
    object_detection_3d = "object_detection_3d"


class ObjectDetectionEnum(enum.IntEnum):
    object_class = 0
    truncated = 1
    occluded = 2
    alpha = 3
    box_2d_left = 4
    box_2d_top = 5
    box_2d_right = 6
    box_2d_bottom = 7
    height = 8
    width = 9
    length = 10
    x = 11
    y = 12
    z = 13
    rotation_y = 14
