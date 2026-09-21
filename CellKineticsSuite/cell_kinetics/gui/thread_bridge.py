"""Thread-safe handoff from background threads to the Tk main thread.

Calling self.after(...) (or any other Tk/Tcl call) directly from a
background thread is NOT actually safe, even though it often appears to
work: Tk widgets hold reference cycles, so closing/replacing one leaves
cyclic garbage that Python's automatic GC can collect on *whichever thread*
happens to trip its allocation threshold -- commonly a busy worker thread.
If that GC pass finalizes a Tk-linked object from a background thread, Tcl
crashes hard with "Tcl_AsyncDelete: async handler deleted by the wrong
thread". See main.py's gc.disable() for the other half of this fix.

MainThreadDispatcher sidesteps the problem entirely: background threads only
ever touch a plain thread-safe queue.Queue, never Tk. A poller scheduled
with the widget's own .after() -- and therefore always running on the main
thread -- drains the queue and actually invokes the callbacks.
"""
import queue


class MainThreadDispatcher:
    def __init__(self, tk_widget):
        self._widget = tk_widget
        self._queue = queue.Queue()

    def post(self, fn):
        """Thread-safe: call from any thread to run fn() on the main thread."""
        self._queue.put(fn)

    def start(self, interval_ms=50):
        """Call once, from the main thread, to begin polling."""
        self._poll(interval_ms)

    def _poll(self, interval_ms):
        while True:
            try:
                fn = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as e:
                print(f"Error in queued main-thread callback: {e}")
        self._widget.after(interval_ms, lambda: self._poll(interval_ms))
