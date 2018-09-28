from collections import deque
from multiprocessing import Process, Queue

from .WebcamStream import WebcamStream

class BufferedWebcamStream(WebcamStream, Process):
    def __init__(self, cache_size=2048, threaded=False):
        self._buffer = Queue()
        self.threaded = threaded

        Process.__init__(self)
        self._delayed = True

    def run(self):
        if self._delayed:
            self._delayed = False
            WebcamStream.__init__(self)

        if not self.threaded:
            raise RuntimeError

        # XXX: infinite loop!
        while True:
            self._read()

    def _read(self):
        ''' Read frame from hardware and cache it into buffer. '''
        frame_nro, timestamp, frame = WebcamStream.read(self)
        self._buffer.put((frame_nro, timestamp, frame))


    def read(self, amount=1):
        ''' Return next frame on buffer. '''

        d = deque()
        for _ in range(amount):
            if not self.threaded:
                self._read()
            d.append(self._buffer.get())
        return d

    def __iter__(self):
        return self

    def __next__(self):
        if not self.threaded:
            self._read()
        return self._buffer.get()
