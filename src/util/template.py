from string import Formatter

class TemplateFormatter(Formatter):

    def check_unused_args(self, used_args, kwargs):
        difference = set(kwargs).difference(used_args)

        if difference :
            raise KeyError(difference)
        
