"""Draws a character with a system font and returns it as a small on/off bitmap (for the Chinese xiangqi piece names).

Needs a running Qt application (QApplication / QGuiApplication) because it uses Qt text rendering.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter

_SUPERSAMPLE = 4
_INK_THRESHOLD = 0.4
_FONT_FAMILIES = ["Microsoft JhengHei", "Microsoft YaHei", "MingLiU", "SimSun", "Noto Sans CJK TC", "PingFang TC"]


def glyph_bitmap_render(character: str, size: int) -> tuple[str, ...]:
    """Bitmap of one character: `size` strings of `size` cells, '#' = ink, '.' = empty."""
    big = size * _SUPERSAMPLE
    image = QImage(big, big, QImage.Format.Format_Grayscale8)
    image.fill(255)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    font = QFont()
    font.setFamilies(_FONT_FAMILIES)
    font.setPixelSize(int(big * 0.92))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(0, 0, 0))
    painter.drawText(QRectF(0, 0, big, big), Qt.AlignmentFlag.AlignCenter.value, character)
    painter.end()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(big, image.bytesPerLine())[:, :big]
    ink = (255.0 - raw.astype(float)).reshape(size, _SUPERSAMPLE, size, _SUPERSAMPLE).mean(axis=(1, 3)) / 255.0
    return tuple("".join("#" if value > _INK_THRESHOLD else "." for value in row) for row in ink)
