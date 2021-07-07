class CallbackRunner:
    def __init__(self, callback=None):
        self.callback = callback
        if self.callback is None:
            self.callback = []
            
    def __call__(self, value, score, model):
        for cal in self.callback:
            getattr(cal, value)(score, model)