"""
启动闪屏模块
现代化品牌启动画面，圆角毛玻璃质感 + 逐点加载动画 + 渐入渐出
支持热更新（放入 update_code 重启即可，无需重新打包）
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve


class LoadingWindow(QWidget):
    """精致品牌闪屏"""
    def __init__(self, message: str = "正在唤醒智核引擎，请稍候"):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(520, 320)

        # 主面板 - 毛玻璃质感
        self.panel = QWidget(self)
        self.panel.setGeometry(10, 10, 500, 300)
        self.panel.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.97),
                    stop:1 rgba(240,244,248,0.97)
                );
                border-radius: 24px;
                border: 1px solid rgba(226,232,240,0.8);
            }
        """)

        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(40, 30, 40, 30)

        # Logo 行
        logo = QHBoxLayout()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emoji = QLabel("🧠")
        emoji.setStyleSheet("font-size: 42px; background: transparent; border: none;")
        title = QLabel("  OmniCore")
        title.setStyleSheet("""
            font-size: 42px; font-weight: 700; color: #1e293b;
            background: transparent; border: none; letter-spacing: 2px;
        """)
        logo.addWidget(emoji)
        logo.addWidget(title)

        # 副标题
        subtitle = QLabel("智核工作站")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 16px; color: #64748b;
            background: transparent; border: none; margin-top: 4px;
        """)

        # 加载文本
        self.status_text = message
        self.loading = QLabel(message)
        self.loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading.setStyleSheet("""
            font-size: 14px; color: #94a3b8;
            background: transparent; border: none; margin-top: 20px;
        """)

        # 引擎标签 - 显示核心功能
        engine = QLabel("⚡ OmniCore · 本地微调 · RAG · Agent · 代码生成")
        engine.setAlignment(Qt.AlignmentFlag.AlignCenter)
        engine.setStyleSheet("""
            font-size: 11px; color: #cbd5e1;
            background: transparent; border: none;
        """)

        layout.addStretch(2)
        layout.addLayout(logo)
        layout.addWidget(subtitle)
        layout.addWidget(self.loading)
        layout.addStretch(1)
        layout.addWidget(engine)

        # 逐点动画
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_dots)
        self._timer.start(500)

        # 淡入
        self._fade_in = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setDuration(400)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_in.start()

        # 居中
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.move((g.width() - self.width()) // 2, (g.height() - self.height()) // 2)

    def _update_dots(self):
        self._dots = (self._dots + 1) % 4
        self.loading.setText(self.status_text + "." * self._dots)

    def finish(self, window):
        self._timer.stop()
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setDuration(250)
        anim.setEasingCurve(QEasingCurve.Type.InCubic)
        anim.finished.connect(self.close)
        anim.start()
