
class Pipeline:

    def batch(self,data,size):
        return[data[x: x + size] for x in range (0,len(data), size)]