"""3D viewport: draws the software-rendered scene, left-drag orbits, wheel zooms, double-click resets.

The still parts (floor, board, pieces standing) are projected once and cached; only the arm is rebuilt every frame.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from core.scene_render import (FaceBatch, SceneCamera, scene_render_batch_join, scene_render_faces, scene_render_order,
                               scene_render_pick_ray)
from core.world_scene import WorldScene


def _polygons(batch: FaceBatch) -> list[QPolygonF]:
    return [QPolygonF([QPointF(x, y) for x, y in corners]) for corners in batch.points.tolist()]


def _colors(batch: FaceBatch) -> list[QColor]:
    return [QColor(r, g, b) for r, g, b in batch.colors.tolist()]


class SceneViewport(QWidget):
    """Shows whatever `scene_provider` returns. Call refresh() after the scene changed."""

    clicked = Signal(object, object)  # world ray (origin, direction) of a click that did not drag

    def __init__(self, scene_provider: Callable[[], WorldScene], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_provider = scene_provider
        self._camera = SceneCamera()
        self._drag_from: QPointF | None = None
        self._press_at: QPointF | None = None
        self._fast = False
        self._cache_key: tuple | None = None
        self._background_image: QImage | None = None
        self._static_batch: FaceBatch | None = None
        self._static_polygons: list[QPolygonF] = []
        self._static_colors: list[QColor] = []
        self.setMinimumSize(520, 380)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_fast(self, fast: bool) -> None:
        """Fast mode skips anti-aliasing while the arm is moving."""
        self._fast = fast

    def refresh(self) -> None:
        """Redraw with the current scene."""
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._drag_from = event.position()
        self._press_at = event.position()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        was_click = self._press_at is not None and (event.position() - self._press_at).manhattanLength() < 5
        self._drag_from = None
        self._press_at = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if was_click and event.button() == Qt.MouseButton.LeftButton:
            origin, direction = scene_render_pick_ray(self._camera, self.width(), self.height(),
                                                      event.position().x(), event.position().y())
            self.clicked.emit(origin, direction)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is None:
            return
        if self._press_at is not None and (event.position() - self._press_at).manhattanLength() >= 5:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        delta = event.position() - self._drag_from
        self._drag_from = event.position()
        self._camera.azimuth_deg -= delta.x() * 0.4
        self._camera.elevation_deg = min(85.0, max(5.0, self._camera.elevation_deg + delta.y() * 0.4))
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self._camera.distance_mm = min(2600.0, max(450.0, self._camera.distance_mm * factor))
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self._camera = SceneCamera()
        self.update()

    def _update_static_cache(self, scene: WorldScene, width: int, height: int) -> None:
        camera = self._camera
        ratio = self.devicePixelRatioF()
        key = (scene.static_key, camera.azimuth_deg, camera.elevation_deg, camera.distance_mm, width, height, ratio)
        if key == self._cache_key:
            return
        self._cache_key = key
        image = QImage(int(width * ratio), int(height * ratio), QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0, 0, width, height)
        backdrop = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        backdrop.setColorAt(0.0, QColor(46, 52, 64))
        backdrop.setColorAt(1.0, QColor(18, 21, 28))
        painter.fillRect(rect, backdrop)
        for layer in scene.background:  # each layer is sorted inside, layers keep their order
            batch = scene_render_faces(layer, camera, width, height)
            polygons, colors = _polygons(batch), _colors(batch)
            for index in scene_render_order(batch.depth).tolist():
                painter.setBrush(colors[index])
                painter.setPen(QPen(colors[index], 0.8))
                painter.drawPolygon(polygons[index])
        painter.end()
        self._background_image = image
        self._static_batch = scene_render_faces(scene.static_objects, camera, width, height)
        self._static_polygons = _polygons(self._static_batch)
        self._static_colors = _colors(self._static_batch)

    def paintEvent(self, event) -> None:  # noqa: N802
        width, height = self.width(), self.height()
        scene = self._scene_provider()
        self._update_static_cache(scene, width, height)
        dynamic = scene_render_faces(scene.dynamic_objects, self._camera, width, height)
        joined = scene_render_batch_join(self._static_batch, dynamic)
        order = scene_render_order(joined.depth).tolist()
        polygons = self._static_polygons + _polygons(dynamic)
        colors = self._static_colors + _colors(dynamic)

        fast = self._fast or self._drag_from is not None
        image = self._background_image.copy()
        pen_painter = QPainter(image)
        pen_painter.setRenderHint(QPainter.RenderHint.Antialiasing, not fast)
        if fast:
            pen_painter.setPen(Qt.PenStyle.NoPen)
        for index in order:
            color = colors[index]
            pen_painter.setBrush(color)
            if not fast:
                pen_painter.setPen(QPen(color, 0.8))
            pen_painter.drawPolygon(polygons[index])
        pen_painter.end()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        frame = QPainterPath()
        frame.addRoundedRect(rect, 18, 18)
        painter.setClipPath(frame)
        painter.drawImage(0, 0, image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipping(False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
        painter.drawPath(frame)
