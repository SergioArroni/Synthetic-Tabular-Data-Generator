


class TTSTTR:
    def __init__(self, tts, ttr):
        self.tts = tts
        self.ttr = ttr

    def get_tts(self):
        return self.tts

    def get_ttr(self):
        return self.ttr

    def get_tts_ttr(self):
        return self.tts, self.ttr

    def set_tts(self, tts):
        self.tts = tts

    def set_ttr(self, ttr):
        self.ttr = ttr

    def set_tts_ttr(self, tts, ttr):
        self.tts = tts
        self.ttr = ttr

    def __str__(self):
        return "TTS: " + str(self.tts) + ", TTR: " + str(self.ttr)