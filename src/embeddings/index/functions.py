from types import FunctionType, MethodType

class Functions:

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.references = None

    def __call__(self, config):

        self.references = []

        functions = []
        for fn in config["functions"]:
            if isinstance(fn, dict):
                fn = fn.copy()
                fn["function"] = self.function(fn["function"])
            else:
                fn = self.function(fn)
            functions.append(fn)
        return functions

    def reset(self):
        
        if self.references:
            for reference in self.references:
                reference.reset()

    def function(self, function):

        if isinstance(function, str):
            parts = function.split(".")
            if hasattr(self.embeddings, parts[0]):
                m = Reference(self.embeddings, parts[0])
                self.references.append(m)
            else:
                module = ".".join(parts[:-1])
                m = __import__(module)
            for comp in parts[1:]:
                m = Reference(m, comp)
                self.references.append(m)
            return m
        return function

class Reference:
    def __init__(self, obj, attribute):
        self.obj = obj
        self.attribute = attribute
        self.inputs = (obj, attribute)
        self.resolved = False
        self.function = None

    def __call__(self, *args):
        if not self.resolved:
            self.obj = self.obj() if isinstance(self.obj, Reference) else self.obj
            self.attribute = self.attribute() if isinstance(self.attribute, Reference) else self.attribute
            self.resolved = True
        attribute = getattr(self.obj, self.attribute)
        if self.function is None:
            self.function = isinstance(attribute, (FunctionType, MethodType)) or (hasattr(attribute, "__call__") and args)
        return attribute(*args) if self.function else attribute

    def reset(self):
        self.obj, self.attribute = self.inputs
        self.resolved = False
