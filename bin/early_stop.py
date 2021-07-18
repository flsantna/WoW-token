from numpy import std


class BreakException(Exception):
    pass


class EarlyStop(object):
    def __init__(self):
        self.list_loss = []
        self.best_model = None

    def get_best_model(self):
        return self.best_model[0]

    def check(self, loss, model, patience=10):
        self.list_loss.append(loss)

        if self.best_model is None:
            self.best_model = [model, loss]

        if self.list_loss[-1] < self.best_model[1]:
            self.best_model = [model, loss]

        if len(self.list_loss) == patience:
            std_value = std(self.list_loss)
            self.list_loss = []
            if std_value == 0:
                raise BreakException()
