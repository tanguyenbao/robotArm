"""Moonsilver palette and the Qt style sheet (same look as the ElderScroll tools hub)."""
from __future__ import annotations

TEXT_PRIMARY = "#EEF1F6"
TEXT_SECONDARY = "#A9B1BE"
TEXT_ENGRAVED_DARK = "#1D2128"
MONO_FONTS = '"Cascadia Mono", "Consolas", monospace'

STYLE_SHEET = f"""
QWidget {{
    font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {TEXT_PRIMARY};
}}
QLabel {{ background: transparent; }}
QLabel#brand {{ font-size: 18px; font-weight: 600; }}
QLabel#heading {{ font-size: 13px; font-weight: 600; color: {TEXT_SECONDARY}; letter-spacing: 1px; }}
QLabel#caption {{ color: {TEXT_SECONDARY}; font-size: 12px; }}
QLabel#status {{ font-size: 14px; font-weight: 600; }}
QLabel#mono {{ font-family: {MONO_FONTS}; font-size: 12px; color: #C9D0DA; }}

QLineEdit {{
    background: rgba(8, 10, 14, 120);
    border: 1px solid rgba(255, 255, 255, 46);
    border-radius: 10px;
    padding: 8px 12px;
    selection-background-color: rgba(201, 208, 218, 120);
}}
QLineEdit:focus {{ border: 1px solid rgba(235, 240, 248, 150); }}

QComboBox {{
    background: rgba(8, 10, 14, 120);
    border: 1px solid rgba(255, 255, 255, 46);
    border-radius: 10px;
    padding: 8px 12px;
}}
QComboBox::drop-down {{ border: none; width: 30px; }}
QComboBox::down-arrow {{
    width: 0px; height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #C9D0DA;
    margin-right: 10px;
}}
QComboBox QAbstractItemView {{
    background: #1A1D24;
    color: {TEXT_PRIMARY};
    border: 1px solid rgba(255, 255, 255, 60);
    selection-background-color: #4A515D;
    outline: 0;
}}

QPlainTextEdit {{
    background: rgba(6, 8, 12, 150);
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 12px;
    padding: 8px;
    font-family: {MONO_FONTS};
    font-size: 12px;
    selection-background-color: rgba(201, 208, 218, 120);
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: rgba(255, 255, 255, 70); border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: rgba(255, 255, 255, 38); }}
QSlider::sub-page:horizontal {{
    border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8D96A3, stop:1 #F4F6FA);
}}
QSlider::handle:horizontal {{
    width: 22px; height: 22px; margin: -9px 0; border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 170);
    background: qradialgradient(cx:0.35, cy:0.3, radius:0.9, fx:0.35, fy:0.3,
                                stop:0 #FFFFFF, stop:0.55 #C9D0DA, stop:1 #8892A0);
}}

QMenu, QToolTip, QDialog, QMessageBox {{ background: #1E222A; color: {TEXT_PRIMARY}; border: 1px solid rgba(255, 255, 255, 50); }}
QAbstractItemView {{ background: #171B22; color: {TEXT_PRIMARY}; selection-background-color: rgba(201, 208, 218, 70); selection-color: #FFFFFF; }}
"""
