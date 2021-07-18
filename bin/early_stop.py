from numpy import std


class BreakException(Exception):
    pass


class EarlyStopping(object):
    def __init__(self):
        self.list_loss = []
        self.list_model = []
        self.best_model = None
        self.last_model = None

    def get_last_model(self):
        return self.list_model

    def get_best_model(self):
        return self.best_model[0]

    def to_break(self, loss, model, patience=10):
        self.list_loss.append(loss)

        if len(self.list_loss) == 1 and self.best_model is None:
            self.best_model = [model, self.list_loss[0]]

        if self.best_model[1] > loss:
            self.best_model = [model, loss]

        if len(self.list_loss) == patience:
            std_value = std(self.list_loss)
            self.list_loss = []
            if std_value == 0:
                self.last_model = model
                raise BreakException()
