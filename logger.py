import sys
 
class Logger:
 
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
 
    def flush(self):
        self.console.flush()
        self.file.flush()
