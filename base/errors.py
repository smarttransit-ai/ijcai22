class ImplementationError(ValueError):
    def __init__(self, *args):
        super(ImplementationError, self).__init__(args)
