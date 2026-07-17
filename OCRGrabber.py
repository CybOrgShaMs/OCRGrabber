# ┌───────────────────────────────────────────────────────────────────────────┐
# │ OCRGrabber                                                               │
# │ Lightweight desktop OCR overlay                                           │
# │                                                                           │
# │ Author     : CyborgShams                                                  │
# │ GitHub     : https://github.com/CybOrgShaMs                               │
# │ Repository : https://github.com/CybOrgShaMs/OCRGrabber                   │
# │ License    : MIT                                                          │
# └───────────────────────────────────────────────────────────────────────────┘


import sys
import signal
from PyQt6 import QtCore, QtGui, QtWidgets
from PIL import Image
import pytesseract
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Palette & shared stylesheet
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG      = "#0f1117"
PANEL_BG     = "#171b26"
BORDER_COLOR = "#2a2f3d"
ACCENT       = "#00e5a0"
ACCENT_DIM   = "#00a870"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED   = "#636a82"
DANGER       = "#ff4d6a"
WARNING      = "#f0a500"
APP_FONT     = "Consolas, 'Courier New', monospace"

GLOBAL_QSS = f"""
QWidget {{
    font-family: {APP_FONT};
    font-size: 12px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QToolTip {{
    background: {PANEL_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    padding: 4px 8px;
    font-size: 11px;
}}
"""

SPINBOX_STYLE = f"""
    QSpinBox, QDoubleSpinBox {{
        background: {DARK_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
        padding: 2px 6px;
        color: {TEXT_PRIMARY};
        min-height: 24px;
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background: {BORDER_COLOR}; border: none; width: 16px;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {ACCENT_DIM};
    }}
"""

COMBO_STYLE = f"""
    QComboBox {{
        background: {DARK_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
        padding: 2px 8px;
        color: {TEXT_PRIMARY};
        min-height: 24px;
    }}
    QComboBox:focus {{ border-color: {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{
        background: {PANEL_BG};
        border: 1px solid {BORDER_COLOR};
        selection-background-color: {ACCENT};
        selection-color: {DARK_BG};
    }}
"""

CHECKBOX_STYLE = f"""
    QCheckBox {{ spacing: 8px; color: {TEXT_PRIMARY}; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {BORDER_COLOR};
        border-radius: 3px;
        background: {DARK_BG};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT};
    }}
"""

SLIDER_STYLE = f"""
    QSlider::groove:horizontal {{
        height: 4px; background: {BORDER_COLOR}; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT}; border: none;
        width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
"""

LABEL_STYLE   = f"color: {TEXT_MUTED}; font-size: 11px;"
SECTION_STYLE = (f"color: {ACCENT}; font-size: 10px; "
                 f"font-weight: 700; letter-spacing: 1.5px;")


def btn_style(color=ACCENT, text_color=DARK_BG, w=None, h=28):
    w_str = f"min-width:{w}px;" if w else ""
    return f"""
        QPushButton {{
            background: {color}; color: {text_color};
            border: none; border-radius: 5px;
            padding: 0 12px; {w_str} min-height: {h}px;
            font-weight: 600; font-size: 11px; letter-spacing: 0.5px;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 {color}ee, stop:1 {color}bb);
        }}
        QPushButton:pressed  {{ background: {color}88; }}
        QPushButton:disabled {{ background: {BORDER_COLOR}; color: {TEXT_MUTED}; }}
    """


def ghost_btn_style(color=TEXT_MUTED):
    return f"""
        QPushButton {{
            background: transparent; color: {color};
            border: 1px solid {BORDER_COLOR}; border-radius: 5px;
            padding: 0 10px; min-height: 26px; font-size: 11px;
        }}
        QPushButton:hover  {{ border-color: {color}; color: {TEXT_PRIMARY}; }}
        QPushButton:pressed {{ background: {BORDER_COLOR}; }}
    """


def make_label(text, style=LABEL_STYLE):
    lbl = QtWidgets.QLabel(text)
    lbl.setStyleSheet(style)
    return lbl


def make_section(text):
    lbl = QtWidgets.QLabel(text.upper())
    lbl.setStyleSheet(SECTION_STYLE)
    return lbl


def h_rule():
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setStyleSheet(f"border: none; border-top: 1px solid {BORDER_COLOR};")
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Transparent viewport (the "see-through" center region)
#
# Two modes:
#   • LIVE  (default) — fully transparent, the desktop shows through.
#   • PREVIEW          — shows the exact pixels that were last captured,
#                         stretched to fill the box, with detected words
#                         highlighted on top so the user can see exactly
#                         what OCR read and where.
# ─────────────────────────────────────────────────────────────────────────────
class TransparentViewport(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self._preview_pixmap = None   # QPixmap of the last captured region
        self._word_boxes     = []     # list of (x, y, w, h, text, conf) in pixmap-space

    # ── Public API ──────────────────────────────────────────────────────────
    def set_preview(self, pixmap: QtGui.QPixmap):
        """Show exactly what was captured, filling the whole box."""
        self._preview_pixmap = pixmap
        self._word_boxes = []
        self.update()

    def set_word_boxes(self, boxes):
        """boxes: list of (x, y, w, h, text, conf) in the preview pixmap's
        own pixel coordinate space."""
        self._word_boxes = boxes or []
        self.update()

    def clear_preview(self):
        """Return to fully see-through / live mode."""
        if self._preview_pixmap is not None or self._word_boxes:
            self._preview_pixmap = None
            self._word_boxes = []
            self.update()

    def has_preview(self):
        return self._preview_pixmap is not None

    # ── Paint ───────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return  # stay fully transparent — desktop shows through

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        target = QtCore.QRectF(self.rect())
        pm     = self._preview_pixmap

        # Stretch the captured snapshot to fill the box exactly — this is
        # what guarantees "everything inside the box" is visible, even if
        # the widget was resized slightly since the capture was taken.
        painter.drawPixmap(target, pm, QtCore.QRectF(pm.rect()))

        if self._word_boxes and pm.width() > 0 and pm.height() > 0:
            sx = target.width()  / pm.width()
            sy = target.height() / pm.height()

            fill = QtGui.QColor(ACCENT)
            fill.setAlpha(55)
            pen = QtGui.QPen(QtGui.QColor(ACCENT))
            pen.setWidthF(1.4)

            for (x, y, w, h, text, conf) in self._word_boxes:
                rect = QtCore.QRectF(x * sx, y * sy, w * sx, h * sy)
                painter.setPen(pen)
                painter.setBrush(fill)
                painter.drawRoundedRect(rect, 2, 2)

        # Small legend so it's clear this is a frozen preview, not live desktop
        painter.setPen(QtGui.QColor(ACCENT))
        painter.setFont(QtGui.QFont(APP_FONT, 9, QtGui.QFont.Weight.Bold))
        label = f"PREVIEW · {len(self._word_boxes)} words detected" \
            if self._word_boxes else "PREVIEW"
        painter.drawText(target.adjusted(6, 4, -6, -4),
                          QtCore.Qt.AlignmentFlag.AlignBottom |
                          QtCore.Qt.AlignmentFlag.AlignLeft, label)


# ─────────────────────────────────────────────────────────────────────────────
# OCR worker — runs in a separate thread so the UI stays live
# ─────────────────────────────────────────────────────────────────────────────
class OCRWorker(QtCore.QThread):
    finished    = QtCore.pyqtSignal(str)
    error       = QtCore.pyqtSignal(str)
    progress    = QtCore.pyqtSignal(str)
    boxes_ready = QtCore.pyqtSignal(list)   # [(x, y, w, h, text, conf), ...]

    def __init__(self, pixmap: QtGui.QPixmap, settings: dict):
        super().__init__()
        self.pixmap   = pixmap
        self.settings = settings

    def run(self):
        try:
            self.progress.emit("Converting image")
            img = self.pixmap.toImage().convertToFormat(
                QtGui.QImage.Format.Format_RGBA8888
            )
            ptr = img.bits()
            ptr.setsize(img.width() * img.height() * 4)
            arr = (np.frombuffer(ptr, dtype=np.uint8)
                   .reshape((img.height(), img.width(), 4))
                   .copy())

            self.progress.emit("Preprocessing")
            gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)

            if self.settings.get("invert", False):
                gray = cv2.bitwise_not(gray)

            scale = self.settings.get("scale", 2)
            if scale > 1:
                gray = cv2.resize(
                    gray,
                    (gray.shape[1] * scale, gray.shape[0] * scale),
                    interpolation=cv2.INTER_CUBIC
                )

            if self.settings.get("denoise", True):
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

            if self.settings.get("despeckle", True):
                # Median blur kills isolated salt-and-pepper pixels (dust,
                # compression artifacts, thin UI hairlines) that Otsu/adaptive
                # thresholding would otherwise turn into little dark blobs —
                # which Tesseract then happily "reads" as stray punctuation
                # or letters.
                gray = cv2.medianBlur(gray, 3)

            mode = self.settings.get("threshold", "combined")
            if mode == "otsu":
                _, processed = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            elif mode == "adaptive":
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 10
                )
            else:  # combined
                _, otsu = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 10
                )
                processed = cv2.bitwise_and(otsu, adaptive)

            if self.settings.get("despeckle", True):
                # Remove any leftover isolated dark speckles that survived
                # thresholding (morphological opening on the inverted image,
                # so we're pruning small *foreground* blobs, not eating into
                # the background). A 2x2 kernel is small enough to leave real
                # letter strokes intact once the image has been upscaled.
                inverted = cv2.bitwise_not(processed)
                kernel   = np.ones((2, 2), np.uint8)
                opened   = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
                processed = cv2.bitwise_not(opened)

            pil_img = Image.fromarray(processed)

            psm    = self.settings.get("psm", "3")
            lang   = self.settings.get("lang", "eng")
            config = f"--psm {psm} --oem 3 -c preserve_interword_spaces=1"
            min_conf = self.settings.get("min_confidence", 50)

            self.progress.emit("Running OCR")
            data = pytesseract.image_to_data(
                pil_img, lang=lang, config=config,
                output_type=pytesseract.Output.DICT
            )

            # Build the final text AND the highlight boxes from the same
            # confidence-filtered word list, so a noise speck that's too
            # low-confidence to trust can't sneak into the transcript OR
            # get a highlight box drawn around it.
            n = len(data.get("text", []))
            kept = []
            for i in range(n):
                raw_word = data["text"][i]
                if raw_word is None:
                    continue
                word = raw_word.strip()
                if not word:
                    continue
                try:
                    conf = float(data["conf"][i])
                except (ValueError, TypeError):
                    conf = -1
                if conf < 0:
                    continue

                # Lone symbols (a single character that isn't a letter or
                # digit) are the classic false-positive: a scratch, a border
                # pixel, or a bit of icon artwork read as "." or "'" or "-".
                # Demand a much higher confidence before trusting one.
                required_conf = min_conf
                if len(word) == 1 and not word.isalnum():
                    required_conf = max(min_conf, 75)
                if conf < required_conf:
                    continue

                kept.append({
                    "block": data["block_num"][i],
                    "par":   data["par_num"][i],
                    "line":  data["line_num"][i],
                    "word":  data["word_num"][i],
                    "x": data["left"][i]   / scale,
                    "y": data["top"][i]    / scale,
                    "w": data["width"][i]  / scale,
                    "h": data["height"][i] / scale,
                    "text": word,
                    "conf": conf,
                })

            kept.sort(key=lambda k: (k["block"], k["par"], k["line"], k["word"]))

            lines_out, boxes = [], []
            cur_key, cur_line_words, prev_block_par = None, [], None
            for k in kept:
                key = (k["block"], k["par"], k["line"])
                if key != cur_key:
                    if cur_line_words:
                        lines_out.append(" ".join(cur_line_words))
                    cur_line_words = []
                    cur_key = key
                    bp = (k["block"], k["par"])
                    if prev_block_par is not None and bp != prev_block_par:
                        lines_out.append("")   # blank line between paragraphs
                    prev_block_par = bp
                cur_line_words.append(k["text"])
                boxes.append((k["x"], k["y"], k["w"], k["h"], k["text"], k["conf"]))
            if cur_line_words:
                lines_out.append(" ".join(cur_line_words))

            self.finished.emit("\n".join(lines_out))
            self.boxes_ready.emit(boxes)

        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Settings window — independent top-level window (no QDialog parent trick)
# ─────────────────────────────────────────────────────────────────────────────
class SettingsWindow(QtWidgets.QWidget):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self.setWindowTitle("OCRGrabber · Settings")
        self.setMinimumWidth(400)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window |
            QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setStyleSheet(
            f"QWidget {{ background: {DARK_BG}; }}"
            + SPINBOX_STYLE + COMBO_STYLE + CHECKBOX_STYLE + SLIDER_STYLE
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        # Title
        title = QtWidgets.QLabel("SETTINGS")
        title.setStyleSheet(
            f"color:{ACCENT}; font-size:13px; font-weight:700; letter-spacing:2px;"
        )
        root.addWidget(title)
        root.addWidget(h_rule())

        # ── OVERLAY ─────────────────────────────────────────────────────────
        root.addWidget(make_section("Overlay"))

        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)

        g.addWidget(make_label("Width"), 0, 0)
        self.w_spin = QtWidgets.QSpinBox()
        self.w_spin.setRange(200, 3840)
        self.w_spin.setValue(overlay.width())
        g.addWidget(self.w_spin, 0, 1)

        g.addWidget(make_label("Height"), 0, 2)
        self.h_spin = QtWidgets.QSpinBox()
        self.h_spin.setRange(150, 2160)
        self.h_spin.setValue(overlay.height())
        g.addWidget(self.h_spin, 0, 3)

        g.addWidget(make_label("H Gap"), 1, 0)
        self.gx_spin = QtWidgets.QSpinBox()
        self.gx_spin.setRange(0, 600)
        self.gx_spin.setValue(overlay.border_gap_x)
        g.addWidget(self.gx_spin, 1, 1)

        g.addWidget(make_label("V Gap"), 1, 2)
        self.gy_spin = QtWidgets.QSpinBox()
        self.gy_spin.setRange(0, 600)
        self.gy_spin.setValue(overlay.border_gap_y)
        g.addWidget(self.gy_spin, 1, 3)

        g.addWidget(make_label("Border px"), 2, 0)
        self.border_spin = QtWidgets.QSpinBox()
        self.border_spin.setRange(2, 32)
        self.border_spin.setValue(overlay.border_thickness)
        g.addWidget(self.border_spin, 2, 1)

        g.addWidget(make_label("Opacity %"), 2, 2)
        self.opacity_spin = QtWidgets.QSpinBox()
        self.opacity_spin.setRange(10, 100)
        self.opacity_spin.setValue(int(overlay.border_opacity * 100))
        g.addWidget(self.opacity_spin, 2, 3)

        root.addLayout(g)
        root.addWidget(h_rule())

        # ── OCR ─────────────────────────────────────────────────────────────
        root.addWidget(make_section("OCR"))

        og = QtWidgets.QGridLayout()
        og.setHorizontalSpacing(12)
        og.setVerticalSpacing(8)

        og.addWidget(make_label("Page Seg Mode"), 0, 0)
        self.psm_combo = QtWidgets.QComboBox()
        self.psm_combo.addItems([
            "1 - Auto + OSD",
            "3 - Auto (default)",
            "4 - Single column",
            "6 - Uniform block",
            "7 - Single line",
            "11 - Sparse text",
            "13 - Raw line",
        ])
        psm_map = {"1": 0, "3": 1, "4": 2, "6": 3, "7": 4, "11": 5, "13": 6}
        self.psm_combo.setCurrentIndex(
            psm_map.get(str(overlay.ocr_settings.get("psm", "3")), 1)
        )
        og.addWidget(self.psm_combo, 0, 1, 1, 3)

        og.addWidget(make_label("Language"), 1, 0)
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItems([
            "eng", "fra", "deu", "spa", "ita",
            "por", "chi_sim", "jpn", "kor"
        ])
        self.lang_combo.setCurrentText(
            overlay.ocr_settings.get("lang", "eng")
        )
        og.addWidget(self.lang_combo, 1, 1, 1, 3)

        og.addWidget(make_label("Threshold"), 2, 0)
        self.thresh_combo = QtWidgets.QComboBox()
        self.thresh_combo.addItems(["combined", "otsu", "adaptive"])
        self.thresh_combo.setCurrentText(
            overlay.ocr_settings.get("threshold", "combined")
        )
        og.addWidget(self.thresh_combo, 2, 1)

        og.addWidget(make_label("Scale x"), 2, 2)
        self.scale_spin = QtWidgets.QSpinBox()
        self.scale_spin.setRange(1, 4)
        self.scale_spin.setValue(overlay.ocr_settings.get("scale", 2))
        og.addWidget(self.scale_spin, 2, 3)

        og.addWidget(make_label("Min Confidence %"), 3, 0)
        self.min_conf_spin = QtWidgets.QSpinBox()
        self.min_conf_spin.setRange(0, 100)
        self.min_conf_spin.setValue(overlay.ocr_settings.get("min_confidence", 50))
        self.min_conf_spin.setToolTip(
            "Words below this confidence are dropped from both the text\n"
            "and the highlight boxes — raise this if noise is being read\n"
            "as letters, lower it if real text is getting cut."
        )
        og.addWidget(self.min_conf_spin, 3, 1)

        root.addLayout(og)

        cr = QtWidgets.QHBoxLayout()
        self.chk_denoise = QtWidgets.QCheckBox("Denoise")
        self.chk_denoise.setChecked(overlay.ocr_settings.get("denoise", True))
        self.chk_invert = QtWidgets.QCheckBox("Invert (light-on-dark)")
        self.chk_invert.setChecked(overlay.ocr_settings.get("invert", False))
        self.chk_despeckle = QtWidgets.QCheckBox("Despeckle")
        self.chk_despeckle.setChecked(overlay.ocr_settings.get("despeckle", True))
        self.chk_despeckle.setToolTip(
            "Removes tiny noise specks before OCR so they don't get\n"
            "misread as stray letters or punctuation."
        )
        cr.addWidget(self.chk_denoise)
        cr.addWidget(self.chk_invert)
        cr.addWidget(self.chk_despeckle)
        cr.addStretch()
        root.addLayout(cr)
        root.addWidget(h_rule())

        # ── AUTO CAPTURE ────────────────────────────────────────────────────
        root.addWidget(make_section("Auto Capture"))
        ar = QtWidgets.QHBoxLayout()
        self.chk_auto = QtWidgets.QCheckBox("Enabled")
        self.chk_auto.setChecked(overlay.auto_capture_enabled)
        ar.addWidget(self.chk_auto)
        ar.addWidget(make_label("  Interval (s)"))
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(1, 300)
        self.interval_spin.setValue(overlay.auto_capture_interval)
        ar.addWidget(self.interval_spin)
        ar.addStretch()
        root.addLayout(ar)
        root.addWidget(h_rule())

        # ── Buttons ─────────────────────────────────────────────────────────
        br = QtWidgets.QHBoxLayout()
        br.addStretch()
        reset_btn = QtWidgets.QPushButton("Reset Defaults")
        reset_btn.setStyleSheet(ghost_btn_style())
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setStyleSheet(btn_style())
        apply_btn.setFixedWidth(90)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setStyleSheet(ghost_btn_style())
        close_btn.setFixedWidth(70)
        br.addWidget(reset_btn)
        br.addWidget(apply_btn)
        br.addWidget(close_btn)
        root.addLayout(br)

        apply_btn.clicked.connect(self._apply)
        close_btn.clicked.connect(self.close)
        reset_btn.clicked.connect(self._reset_defaults)

        # Live preview as values change
        for w in (self.w_spin, self.h_spin, self.gx_spin,
                  self.gy_spin, self.border_spin):
            w.valueChanged.connect(self._live_resize)
        self.opacity_spin.valueChanged.connect(self._live_opacity)

    def _live_resize(self):
        o = self.overlay
        o.border_thickness = self.border_spin.value()
        o.border_gap_x     = self.gx_spin.value()
        o.border_gap_y     = self.gy_spin.value()
        o.resize(self.w_spin.value(), self.h_spin.value())
        o.update_layout_margins()
        # Box geometry changed — any frozen preview no longer matches
        # what's underneath, so drop back to the live see-through view.
        o.viewport.clear_preview()
        o.update()

    def _live_opacity(self):
        self.overlay.border_opacity = self.opacity_spin.value() / 100.0
        self.overlay.update()

    def _psm_value(self):
        return self.psm_combo.currentText().split(" ")[0]

    def _apply(self):
        self._live_resize()
        self._live_opacity()
        self.overlay.ocr_settings.update({
            "psm":            self._psm_value(),
            "lang":           self.lang_combo.currentText(),
            "threshold":      self.thresh_combo.currentText(),
            "scale":          self.scale_spin.value(),
            "denoise":        self.chk_denoise.isChecked(),
            "invert":         self.chk_invert.isChecked(),
            "despeckle":      self.chk_despeckle.isChecked(),
            "min_confidence": self.min_conf_spin.value(),
        })
        self.overlay.auto_capture_enabled  = self.chk_auto.isChecked()
        self.overlay.auto_capture_interval = self.interval_spin.value()
        self.overlay.apply_auto_capture()
        self.close()

    def _reset_defaults(self):
        self.w_spin.setValue(600)
        self.h_spin.setValue(400)
        self.gx_spin.setValue(0)
        self.gy_spin.setValue(50)
        self.border_spin.setValue(8)
        self.opacity_spin.setValue(80)
        self.psm_combo.setCurrentIndex(1)
        self.lang_combo.setCurrentText("eng")
        self.thresh_combo.setCurrentText("combined")
        self.scale_spin.setValue(2)
        self.chk_denoise.setChecked(True)
        self.chk_invert.setChecked(False)
        self.chk_despeckle.setChecked(True)
        self.min_conf_spin.setValue(50)
        self.chk_auto.setChecked(False)
        self.interval_spin.setValue(5)


# ─────────────────────────────────────────────────────────────────────────────
# Results window — independent top-level, always accessible
# ─────────────────────────────────────────────────────────────────────────────
class CaptureWindow(QtWidgets.QWidget):
    def __init__(self, overlay):
        super().__init__()          # NO parent — prevents hide-with-overlay bug
        self.overlay = overlay
        self.setWindowTitle("OCRGrabber · Results")
        self.setMinimumSize(420, 340)
        self.resize(480, 440)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window |
            QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setStyleSheet(f"""
            QWidget   {{ background: {DARK_BG}; color: {TEXT_PRIMARY};
                         font-family: {APP_FONT}; font-size: 12px; }}
            QTextEdit {{
                background: {PANEL_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 10px;
                color: {TEXT_PRIMARY};
                selection-background-color: {ACCENT};
                selection-color: {DARK_BG};
            }}
            QScrollBar:vertical {{
                background: {DARK_BG}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR}; border-radius: 4px; min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Header row
        hdr = QtWidgets.QHBoxLayout()
        hdr.addWidget(QtWidgets.QLabel(
            "RESULTS",
            styleSheet=(
                f"color:{ACCENT}; font-size:12px; "
                f"font-weight:700; letter-spacing:2px;"
            )
        ))
        hdr.addStretch()
        self.char_lbl = make_label("0 chars")
        hdr.addWidget(self.char_lbl)
        root.addLayout(hdr)

        # Text area
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setPlaceholderText("Captured text will appear here…")
        self.text_edit.textChanged.connect(self._update_count)
        root.addWidget(self.text_edit)

        # Status bar
        self.status_lbl = QtWidgets.QLabel("Ready")
        self.status_lbl.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:10px; padding: 2px 0;"
        )
        root.addWidget(self.status_lbl)

        # Action buttons
        br = QtWidgets.QHBoxLayout()
        self.copy_btn  = QtWidgets.QPushButton("Copy All")
        self.copy_btn.setStyleSheet(btn_style(ACCENT))
        self.save_btn  = QtWidgets.QPushButton("Save…")
        self.save_btn.setStyleSheet(ghost_btn_style(TEXT_MUTED))
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.setStyleSheet(ghost_btn_style(DANGER))
        br.addWidget(self.copy_btn)
        br.addWidget(self.save_btn)
        br.addStretch()
        br.addWidget(self.clear_btn)
        root.addLayout(br)

        self.copy_btn.clicked.connect(self._copy_all)
        self.save_btn.clicked.connect(self._save)
        self.clear_btn.clicked.connect(self.text_edit.clear)

        # Spinner
        self._spin_timer  = QtCore.QTimer(self)
        self._spin_timer.timeout.connect(self._tick)
        self._spin_chars  = list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        self._spin_idx    = 0
        self._spin_msg    = ""

    # ── Char count ───────────────────────────────────────────────────────────
    def _update_count(self):
        n = len(self.text_edit.toPlainText())
        self.char_lbl.setText(f"{n:,} chars")

    # ── Copy / Save ──────────────────────────────────────────────────────────
    def _copy_all(self):
        txt = self.text_edit.toPlainText()
        if txt:
            QtWidgets.QApplication.clipboard().setText(txt)
            self._flash("Copied to clipboard ✓", ACCENT)

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save OCR Text", "",
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text_edit.toPlainText())
            self._flash(f"Saved → {path}", ACCENT)

    # ── Spinner / status ─────────────────────────────────────────────────────
    def set_busy(self, msg="Processing"):
        self._spin_msg = msg
        self._spin_idx = 0
        self.copy_btn.setEnabled(False)
        self._spin_timer.start(80)

    def set_idle(self):
        self._spin_timer.stop()
        self.copy_btn.setEnabled(True)

    def _tick(self):
        ch = self._spin_chars[self._spin_idx % len(self._spin_chars)]
        self.status_lbl.setText(f"{ch}  {self._spin_msg}…")
        self.status_lbl.setStyleSheet(
            f"color:{WARNING}; font-size:10px; padding:2px 0;"
        )
        self._spin_idx += 1

    def _flash(self, msg, color=ACCENT):
        self.set_idle()
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(
            f"color:{color}; font-size:10px; padding:2px 0;"
        )
        QtCore.QTimer.singleShot(3500, self._reset_status)

    def _reset_status(self):
        self.status_lbl.setText("Ready")
        self.status_lbl.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:10px; padding:2px 0;"
        )

    # ── Called by OverlayUI ──────────────────────────────────────────────────
    def set_text(self, text: str):
        self.set_idle()
        self.text_edit.setPlainText(text)
        self._flash("OCR complete ✓", ACCENT)

    def set_error(self, msg: str):
        self.set_idle()
        self._flash(f"Error: {msg}", DANGER)

    def move_beside_overlay(self):
        og = self.overlay.geometry()
        self.move(og.right() + 12, og.top())


# ─────────────────────────────────────────────────────────────────────────────
# Main overlay widget
# ─────────────────────────────────────────────────────────────────────────────
class OverlayUI(QtWidgets.QWidget):

    LOG_DEBUG     = "\033[96m[DEBUG]\033[0m"
    LOG_INFO      = "\033[92m[INFO]\033[0m"
    LOG_ERROR     = "\033[91m[ERROR]\033[0m"
    LOG_EXCEPTION = "\033[41m[EXCEPTION]\033[0m"

    def __init__(self):
        super().__init__()

        self.border_thickness      = 8
        self.border_gap_x          = 0
        self.border_gap_y          = 50
        self.border_opacity        = 0.80
        self.capture_window        = None
        self.drag_position         = None
        self._ocr_worker           = None
        self.auto_capture_enabled  = False
        self.auto_capture_interval = 5

        self._auto_timer    = QtCore.QTimer(self)
        self._auto_timer.timeout.connect(self.start_ocr)
        self._settings_win  = None   # kept alive here so Qt doesn't GC it

        self.ocr_settings = {
            "psm":            "3",
            "lang":           "eng",
            "threshold":      "combined",
            "scale":          2,
            "denoise":        True,
            "invert":         False,
            "despeckle":      True,
            "min_confidence": 50,
        }

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(600, 400)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.update_layout_margins()
        self.main_layout.setSpacing(0)

        self.viewport = TransparentViewport(self)
        self.main_layout.addWidget(self.viewport)

        self._build_buttons()

    def _build_buttons(self):
        H = 28

        self.btn_results = QtWidgets.QPushButton("◧ Results", self)
        self.btn_results.setFixedSize(90, H)
        self.btn_results.setToolTip("Open / show the captured-text window")
        self.btn_results.setStyleSheet(btn_style(ACCENT))

        self.btn_settings = QtWidgets.QPushButton("⚙ Settings", self)
        self.btn_settings.setFixedSize(90, H)
        self.btn_settings.setStyleSheet(ghost_btn_style())

        self.btn_close = QtWidgets.QPushButton("✕", self)
        self.btn_close.setFixedSize(H, H)
        self.btn_close.setToolTip("Quit")
        self.btn_close.setStyleSheet(btn_style(DANGER, "#fff"))

        self.btn_capture = QtWidgets.QPushButton("⬤ Capture", self)
        self.btn_capture.setFixedSize(95, H)
        self.btn_capture.setToolTip(
            "Run OCR on the current viewport and preview exactly what was read"
        )
        self.btn_capture.setStyleSheet(btn_style(ACCENT))

        self.btn_auto = QtWidgets.QPushButton("⏱ Auto: OFF", self)
        self.btn_auto.setFixedSize(108, H)
        self.btn_auto.setToolTip("Toggle auto-capture (configure interval in Settings)")
        self.btn_auto.setStyleSheet(ghost_btn_style(TEXT_MUTED))

        self.btn_live = QtWidgets.QPushButton("↺ Live View", self)
        self.btn_live.setFixedSize(100, H)
        self.btn_live.setToolTip(
            "Clear the frozen preview and return to the see-through live view"
        )
        self.btn_live.setStyleSheet(ghost_btn_style(TEXT_MUTED))

        self.btn_results.clicked.connect(self.open_capture_window)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_close.clicked.connect(QtWidgets.QApplication.quit)
        self.btn_capture.clicked.connect(self.start_ocr)
        self.btn_auto.clicked.connect(self.toggle_auto_capture)
        self.btn_live.clicked.connect(self.viewport.clear_preview)

    def update_layout_margins(self):
        t = self.border_thickness
        self.main_layout.setContentsMargins(
            t + self.border_gap_x,
            t + self.border_gap_y,
            t + self.border_gap_x,
            t + self.border_gap_y,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sp = 10
        r  = self.rect()
        self.btn_results.move(sp, sp)
        self.btn_close.move(r.width() - self.btn_close.width() - sp, sp)
        self.btn_settings.move(
            self.btn_close.x() - self.btn_settings.width() - sp, sp
        )
        self.btn_capture.move(sp, r.height() - self.btn_capture.height() - sp)
        self.btn_auto.move(
            self.btn_capture.x() + self.btn_capture.width() + sp,
            self.btn_capture.y()
        )
        self.btn_live.move(
            self.btn_auto.x() + self.btn_auto.width() + sp,
            self.btn_capture.y()
        )

    # ── OCR ─────────────────────────────────────────────────────────────────
    def start_ocr(self):
        if self._ocr_worker and self._ocr_worker.isRunning():
            return

        # Ensure the results window is open and visible
        self.open_capture_window()

        # Grab from the screen the overlay is ACTUALLY on — not always the
        # primary screen.
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if not screen:
            return

        screen_geo = screen.geometry()          # logical pixels
        top_left   = self.viewport.mapToGlobal(QtCore.QPoint(0, 0))
        local_pt   = top_left - screen_geo.topLeft()   # logical, relative to THIS screen
        vp_size    = self.viewport.size()

        self.capture_window.set_busy("Capturing")

        # Hide the overlay AND any other of our own windows that might be
        # sitting next to / overlapping the capture region (e.g. the
        # Results window, which is parked just to the right of the box),
        # so nothing but the desktop underneath ever gets captured.
        was_results_visible  = bool(self.capture_window and self.capture_window.isVisible())
        was_settings_visible = bool(self._settings_win and self._settings_win.isVisible())
        self.setVisible(False)
        if was_results_visible:
            self.capture_window.setVisible(False)
        if was_settings_visible:
            self._settings_win.setVisible(False)

        QtWidgets.QApplication.processEvents()
        QtCore.QThread.msleep(200)

        # Grab the WHOLE screen with no rect args — this is unambiguous
        # regardless of DPI/scale settings. Guessing physical-vs-logical
        # coordinates for a partial-rect grab was what caused the capture
        # box to overreach to the right into the Results window.
        full_pixmap = screen.grabWindow(0)

        self.setVisible(True)
        if was_results_visible:
            self.capture_window.setVisible(True)
        if was_settings_visible:
            self._settings_win.setVisible(True)

        if full_pixmap.isNull():
            print(f"{self.LOG_ERROR} Screen grab failed")
            self.capture_window.set_error("Screen grab returned null pixmap")
            return

        # Derive the true logical→pixel scale factor from the pixmap Qt
        # actually handed back, instead of assuming devicePixelRatio()
        # matches whatever coordinate convention grabWindow expects.
        scale_x = full_pixmap.width()  / max(1, screen_geo.width())
        scale_y = full_pixmap.height() / max(1, screen_geo.height())

        crop_x = round(local_pt.x() * scale_x)
        crop_y = round(local_pt.y() * scale_y)
        crop_w = round(vp_size.width()  * scale_x)
        crop_h = round(vp_size.height() * scale_y)

        # Clamp so we never read outside the grabbed pixmap's bounds.
        crop_x = max(0, min(crop_x, full_pixmap.width()))
        crop_y = max(0, min(crop_y, full_pixmap.height()))
        crop_w = max(1, min(crop_w, full_pixmap.width()  - crop_x))
        crop_h = max(1, min(crop_h, full_pixmap.height() - crop_y))

        print(f"{self.LOG_DEBUG} screen={screen.name()} "
              f"scale=({scale_x:.3f},{scale_y:.3f}) "
              f"crop=({crop_x},{crop_y} {crop_w}x{crop_h})")

        pixmap = full_pixmap.copy(crop_x, crop_y, crop_w, crop_h)

        if pixmap.isNull():
            print(f"{self.LOG_ERROR} Screen grab failed")
            self.capture_window.set_error("Screen grab returned null pixmap")
            return

        # Immediately show exactly what was grabbed, filling the whole box,
        # so the user can see everything that's about to be OCR'd — before
        # the (slower) OCR pass even finishes.
        self.viewport.set_preview(pixmap)

        self._ocr_worker = OCRWorker(pixmap, dict(self.ocr_settings))
        self._ocr_worker.finished.connect(self._ocr_done)
        self._ocr_worker.error.connect(self._ocr_error)
        self._ocr_worker.progress.connect(self._ocr_progress)
        self._ocr_worker.boxes_ready.connect(self._ocr_boxes_ready)
        self._ocr_worker.start()

    def _ocr_progress(self, msg):
        if self.capture_window:
            self.capture_window.set_busy(msg)

    def _ocr_done(self, text):
        print(f"{self.LOG_INFO} OCR done — {len(text)} chars")
        if self.capture_window:
            self.capture_window.set_text(text)

    def _ocr_error(self, msg):
        print(f"{self.LOG_EXCEPTION} {msg}")
        if self.capture_window:
            self.capture_window.set_error(msg)

    def _ocr_boxes_ready(self, boxes):
        # Highlight exactly which words were detected, and where, right on
        # top of the frozen preview inside the overlay's box.
        self.viewport.set_word_boxes(boxes)

    # ── Auto capture ─────────────────────────────────────────────────────────
    def toggle_auto_capture(self):
        self.auto_capture_enabled = not self.auto_capture_enabled
        self.apply_auto_capture()

    def apply_auto_capture(self):
        if self.auto_capture_enabled:
            self._auto_timer.start(self.auto_capture_interval * 1000)
            self.btn_auto.setText(f"⏱ {self.auto_capture_interval}s  ■ LIVE")
            self.btn_auto.setStyleSheet(btn_style(WARNING, DARK_BG))
        else:
            self._auto_timer.stop()
            self.btn_auto.setText("⏱ Auto: OFF")
            self.btn_auto.setStyleSheet(ghost_btn_style(TEXT_MUTED))

    # ── Windows ──────────────────────────────────────────────────────────────
    def open_capture_window(self):
        if not self.capture_window:
            self.capture_window = CaptureWindow(self)
            self.capture_window.move_beside_overlay()
        self.capture_window.show()
        self.capture_window.raise_()
        self.capture_window.activateWindow()

    def open_settings(self):
        # Must hold a reference on self — a local var gets GC'd before Qt shows it
        if self._settings_win is None or not self._settings_win.isVisible():
            self._settings_win = SettingsWindow(self)
        self._settings_win.show()
        self._settings_win.raise_()
        self._settings_win.activateWindow()

    def moveEvent(self, event):
        super().moveEvent(event)
        if self.capture_window:
            self.capture_window.move_beside_overlay()

    # ── Paint ────────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        t     = self.border_thickness
        outer = self.rect().adjusted(t//2, t//2, -t//2, -t//2)
        inner = outer.adjusted(
            self.border_gap_x,  self.border_gap_y,
            -self.border_gap_x, -self.border_gap_y
        )

        # Semi-transparent gap fill
        alpha     = int(self.border_opacity * 200)
        gap_color = QtGui.QColor(15, 17, 23, alpha)
        path      = QtGui.QPainterPath()
        path.addRect(QtCore.QRectF(outer))
        path.addRect(QtCore.QRectF(inner))
        painter.fillPath(path, gap_color)

        # Outer neon border
        pen = QtGui.QPen(QtGui.QColor(ACCENT))
        pen.setWidth(t)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(outer)

        # Inner dim border
        pen2 = QtGui.QPen(QtGui.QColor(ACCENT_DIM))
        pen2.setWidth(max(1, t // 3))
        painter.setPen(pen2)
        painter.drawRect(inner)

        # Corner tick marks on inner rect
        tick     = 14
        tick_pen = QtGui.QPen(QtGui.QColor(ACCENT))
        tick_pen.setWidth(2)
        painter.setPen(tick_pen)
        for pt, dx, dy in [
            (inner.topLeft(),     ( 1, 0), (0,  1)),
            (inner.topRight(),    (-1, 0), (0,  1)),
            (inner.bottomLeft(),  ( 1, 0), (0, -1)),
            (inner.bottomRight(), (-1, 0), (0, -1)),
        ]:
            x, y = pt.x(), pt.y()
            painter.drawLine(x, y, x + dx[0]*tick, y + dx[1]*tick)
            painter.drawLine(x, y, x + dy[0]*tick, y + dy[1]*tick)

    # ── Drag ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
            # About to move the box — any frozen preview will no longer
            # correspond to what's underneath once we're done dragging.
            self.viewport.clear_preview()

    def mouseMoveEvent(self, event):
        if (event.buttons() == QtCore.Qt.MouseButton.LeftButton
                and self.drag_position):
            self.move(event.globalPosition().toPoint() - self.drag_position)

    def mouseReleaseEvent(self, event):
        self.drag_position = None


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    overlay = OverlayUI()
    overlay.show()

    sys.exit(app.exec())