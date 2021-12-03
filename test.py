from multiprocessing import Pool
from time import sleep
import time

start = time.time()

def do(waitTime):
    sleep(waitTime*10)
    print("do waitTime:{} 開始から{}秒経過".format(waitTime, time.time() - start))
    return [waitTime]

if __name__ == '__main__':
    waitTimes = [3,2,1]
    with Pool(10) as p:
        for result in p.imap_unordered(do,waitTimes):
            print("result waitTime:{} 開始から{}秒経過".format(result,time.time() - start))
