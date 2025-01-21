import time
import shutil

def add_front_0(string, digits=2, zero = "0"):
    ret = ""
    string = str(string)
    for i in range(digits-len(string)):
        ret += zero
    ret += string
    return ret

class ProgressBar:
    def __init__(self, total, style="=", start=0):
        self.start_time = time.perf_counter()
        self.total = total
        self.start = start
        self.current = start
        try:
            self.width = shutil.get_terminal_size()[0] - 15
        except:
            self.width = 19
        self.style = style

    def add(self, increment=1):
        self.current += increment
        if self.current == self.total:
            self.finish()
        else:
            self.update()

    def restart(self,total=None):
        self.current = self.start
        if total is not None:
            self.total = total

    def finish(self):
        self.update(end="\n")
        print("Completed in {} seconds".format(round(time.perf_counter() - self.start_time), 2))

    def update(self, end="\r"):
        progress = int(self.current * 100 // self.total)
        progress_scaled = int(progress * self.width //100)
        percentage = "|{}%".format(add_front_0(progress, digits=3, zero = " "))
        bar = "|{}>".format(self.style * progress_scaled)
        blank = " " * (self.width - len(bar))
        print("{}{}{}".format(bar, blank, percentage), end = end)

