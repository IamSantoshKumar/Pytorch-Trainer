class CallbackRunner:
    def __init__(self, model, callback=None):
        self.model = model
        self.callback = callback
        if self.callback is None:
            self.callback = []
            
    def __call__(self, value, score, **kwargs):
        for cal in self.callback:
            getattr(cal, value)(self.model, score, **kwargs)

class Callback:
    def on_epoch_start(self, model, **kwargs):
        return

    def on_epoch_end(self, model, **kwargs):
        return

    def on_train_epoch_start(self, model, **kwargs):
        return

    def on_train_epoch_end(self, model, **kwargs):
        return

    def on_valid_epoch_start(self, model, **kwargs):
        return

    def on_valid_epoch_end(self, model, **kwargs):
        return

    def on_train_step_start(self, model, **kwargs):
        return

    def on_train_step_end(self, model, **kwargs):
        return

    def on_valid_step_start(self, model, **kwargs):
        return

    def on_valid_step_end(self, model, **kwargs):
        return

    def on_test_step_start(self, model, **kwargs):
        return

    def on_test_step_end(self, model, **kwargs):
        return

    def on_train_start(self, model, **kwargs):
        return

    def on_train_end(self, model, **kwargs):
        return
