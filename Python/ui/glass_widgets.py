"""Moonsilver window background, glass cards and brushed-metal buttons, painted by hand."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient, QFont
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QPushButton, QWidget

from ui.theme import TEXT_ENGRAVED_DARK, TEXT_PRIMARY

_METAL_STOPS = {
    "silver": [(0.0, "#FFFFFF"), (0.47, "#E3E7ED"), (0.5, "#C3CAD4"), (1.0, "#F1F3F7")],
    "black": [(0.0, "#6C727B"), (0.47, "#2B2F36"), (0.5, "#14171B"), (1.0, "#2D3239")],
}
_RIM_STOPS = {"silver": ("#FFFFFF", "#7C8695"), "black": ("#C4CCD8", "#06070A")}


class MoonsilverBackground(QWidget):
    """Window background: dark silver gradient with soft moonlight glows."""

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        rect = QRectF(self.rect())
        base = QLinearGradient(rect.topLeft(), rect.bottomRight())
        base.setColorAt(0.0, QColor(58, 64, 78))
        base.setColorAt(1.0, QColor(17, 19, 25))
        painter.fillRect(rect, base)
        moonlight = QRadialGradient(rect.width() * 0.18, -rect.height() * 0.05, rect.width() * 0.9)
        moonlight.setColorAt(0.0, QColor(214, 222, 236, 80))
        moonlight.setColorAt(1.0, QColor(214, 222, 236, 0))
        painter.fillRect(rect, moonlight)
        glow = QRadialGradient(rect.width() * 0.95, rect.height() * 1.02, rect.width() * 0.7)
        glow.setColorAt(0.0, QColor(160, 172, 192, 60))
        glow.setColorAt(1.0, QColor(160, 172, 192, 0))
        painter.fillRect(rect, glow)


class GlassCard(QFrame):
    """Frosted panel: a translucent white wash with a lit top-left edge."""

    def __init__(self, parent: QWidget | None = None, radius: float = 18.0) -> None:
        super().__init__(parent)
        self._radius = radius

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        wash = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        wash.setColorAt(0.0, QColor(255, 255, 255, 38))
        wash.setColorAt(1.0, QColor(255, 255, 255, 14))
        painter.fillPath(path, wash)
        edge = QLinearGradient(rect.topLeft(), rect.bottomRight())
        edge.setColorAt(0.0, QColor(255, 255, 255, 130))
        edge.setColorAt(0.5, QColor(255, 255, 255, 34))
        edge.setColorAt(1.0, QColor(255, 255, 255, 60))
        painter.setPen(QPen(edge, 1.0))
        painter.drawPath(path)


class MetalButton(QPushButton):
    """Brushed-metal button. variant 'silver' is the primary action, 'black' the secondary one."""

    def __init__(self, text: str, variant: str = "silver", height: int = 40, radius: float = 13.0,
                 point_size: float = 10.0, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._variant = variant
        self._radius = radius
        self._point_size = point_size
        self.setFixedHeight(height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 150))
        self.setGraphicsEffect(shadow)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0 if self.isEnabled() else 0.45)
        is_silver = self._variant == "silver"
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        path = QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)

        body = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        for position, hex_color in _METAL_STOPS[self._variant]:
            color = QColor(hex_color)
            if self.isDown():
                color = color.darker(118)
            elif self.underMouse():
                color = color.lighter(110)
            body.setColorAt(position, color)
        painter.fillPath(path, body)

        painter.save()
        painter.setClipPath(path)
        painter.setPen(QPen(QColor(255, 255, 255, 12 if is_silver else 9), 1.0))
        row = int(rect.top())
        while row < rect.bottom():  # brushed-metal lines
            painter.drawLine(int(rect.left()), row, int(rect.right()), row)
            row += 3
        gloss = QLinearGradient(rect.topLeft(), rect.center())
        gloss.setColorAt(0.0, QColor(255, 255, 255, 70 if is_silver else 36))
        gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), rect.height() * 0.5), gloss)
        painter.restore()

        top_color, bottom_color = _RIM_STOPS[self._variant]
        rim = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        rim.setColorAt(0.0, QColor(top_color))
        rim.setColorAt(1.0, QColor(bottom_color))
        painter.setPen(QPen(rim, 1.2))
        painter.drawPath(path)

        font = QFont(self.font())
        font.setPointSizeF(self._point_size)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        text_rect = rect.translated(0, 1.0 if self.isDown() else 0.0)
        if is_silver:
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(text_rect.translated(0, 1.0), Qt.AlignmentFlag.AlignCenter, self.text())
            painter.setPen(QColor(TEXT_ENGRAVED_DARK))
        else:
            painter.setPen(QColor(0, 0, 0, 210))
            painter.drawText(text_rect.translated(0, -1.0), Qt.AlignmentFlag.AlignCenter, self.text())
            painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.text())
