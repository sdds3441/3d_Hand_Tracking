import threading

counter=0
pre=-1


def timer():
    global counter
    counter += 1

    timers = threading.Timer(1, timer)
    timers.start()
    #print(counter)
    if counter == 5:
        timers.cancel()
        return "꿑"

timer()
while True:
    if pre != counter:
        print(counter)
    pre=counter

    if counter==5:
        break
