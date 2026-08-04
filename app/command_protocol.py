from __future__ import annotations

from string import Formatter
from typing import Any


COMMAND_PLACEHOLDERS = {
    "annotation_path",
    "batch_size",
    "confidence",
    "dataset_id",
    "device",
    "image_directory",
    "image_size",
    "input_height",
    "input_width",
    "max_detections",
    "model_id",
    "nms_iou",
    "output_directory",
    "precision",
    "predictions_path",
    "project_directory",
    "request_path",
    "result_path",
    "warmup",
    "weight_path",
}


class CommandTemplateError(ValueError):
    pass


def validate_command_arguments(arguments: list[str]) -> None:
    if not arguments:
        raise CommandTemplateError("执行命令至少需要一个参数")
    formatter = Formatter()
    for argument in arguments:
        if not argument or "\x00" in argument:
            raise CommandTemplateError("执行命令不能包含空参数或空字符")
        try:
            fields = list(formatter.parse(argument))
        except ValueError as exc:
            raise CommandTemplateError(
                f"命令参数中的大括号不完整: {argument}"
            ) from exc
        for _, field_name, format_spec, conversion in fields:
            if field_name is None:
                continue
            if field_name not in COMMAND_PLACEHOLDERS:
                raise CommandTemplateError(
                    f"不支持的命令占位符: {{{field_name}}}"
                )
            if format_spec or conversion:
                raise CommandTemplateError(
                    f"命令占位符不支持格式转换: {{{field_name}}}"
                )


def command_placeholders(arguments: list[str]) -> set[str]:
    validate_command_arguments(arguments)
    formatter = Formatter()
    return {
        field_name
        for argument in arguments
        for _, field_name, _, _ in formatter.parse(argument)
        if field_name is not None
    }


def render_command(
    executable: str,
    arguments: list[str],
    values: dict[str, Any],
) -> list[str]:
    used = command_placeholders(arguments)
    missing = used.difference(values)
    if missing:
        raise CommandTemplateError(
            f"命令缺少占位符值: {', '.join(sorted(missing))}"
        )
    rendered = [argument.format_map(values) for argument in arguments]
    return [executable, *rendered]
