import json
import queue
import subprocess
import sys
import threading
from pathlib import Path


class OverlayStatus:
    def __init__(self, enabled=True):
        self.enabled = enabled and sys.platform == "darwin"
        self.process = None

    def start(self):
        if not self.enabled or self.process is not None:
            return

        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            stdin=subprocess.PIPE,
            text=True,
        )

    def show(self, text):
        if not self.enabled or self.process is None or self.process.stdin is None:
            return
        if self.process.poll() is not None:
            self.enabled = False
            return

        try:
            self.process.stdin.write(json.dumps({"text": text}) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            self.enabled = False

    def stop(self):
        if not self.enabled or self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write(json.dumps({"stop": True}) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            pass


def run_child():
    try:
        from AppKit import (
            NSApp,
            NSApplication,
            NSBackingStoreBuffered,
            NSColor,
            NSFont,
            NSMakeRect,
            NSPanel,
            NSScreen,
            NSTextField,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowStyleMaskBorderless,
            NSFloatingWindowLevel,
        )
        from Foundation import NSObject, NSTimer
    except ImportError:
        print("Overlay disabled: install PyObjC with `python3 -m pip install pyobjc`.")
        return

    messages = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    threading.Thread(target=read_stdin, daemon=True).start()

    class OverlayDelegate(NSObject):
        def applicationDidFinishLaunching_(self, notification):
            screen_frame = NSScreen.mainScreen().frame()
            width = 430
            height = 130
            margin = 24
            x = screen_frame.size.width - width - margin
            y = screen_frame.size.height - height - margin

            self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, width, height),
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
            )
            self.panel.setLevel_(NSFloatingWindowLevel)
            self.panel.setOpaque_(False)
            self.panel.setBackgroundColor_(NSColor.clearColor())
            self.panel.setIgnoresMouseEvents_(True)
            self.panel.setHidesOnDeactivate_(False)
            self.panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            self.label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
            self.label.setEditable_(False)
            self.label.setSelectable_(False)
            self.label.setBezeled_(False)
            self.label.setDrawsBackground_(True)
            self.label.setBackgroundColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.55))
            self.label.setTextColor_(NSColor.whiteColor())
            self.label.setFont_(NSFont.monospacedSystemFontOfSize_weight_(16, 0.35))
            self.label.setStringValue_("Gesture HUD ready")

            self.panel.contentView().addSubview_(self.label)
            self.panel.orderFrontRegardless()
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.05,
                self,
                "pollMessages:",
                None,
                True,
            )

        def pollMessages_(self, timer):
            while True:
                try:
                    message = messages.get_nowait()
                except queue.Empty:
                    break

                if message.get("stop"):
                    NSApp.terminate_(None)
                elif "text" in message:
                    self.label.setStringValue_(message["text"])
                    self.panel.orderFrontRegardless()

    app = NSApplication.sharedApplication()
    delegate = OverlayDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    if "--child" in sys.argv:
        run_child()
