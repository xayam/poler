import wget
import re
import sys
from os.path import exists
import os
import hashlib


def md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as ff:
        for chunk in iter(lambda: ff.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def main():
    folder = 'E:/Chess'
    syzygy = folder + '/syzygy_endgame/'
    domain = 'https://tablebase.lichess.ovh/tables/standard/'
    folders = ['6-get_wdl', '6-dtz', '3-4-5-dtz', '3-4-5-get_wdl']
    ext = ['get_wdl', 'dtz']
    for f in folders:
        for ex in ext:
            index = folder + '/' + f + '.html'
            if not exists(index):
                index2 = wget.download(domain + f + '/', index)
                print(index2)
            ff = open(index, "r")
            html = ff.read()
            if ex == 'get_wdl':
                rtb = re.findall(r'>(.*?\.rtbw)<', html)
            else:
                rtb = re.findall(r'>(.*?\.rtbz)<', html)
            ff.close()
            os.remove(index)
            for wdl in rtb:
                print("\n" + f + '/' + wdl)
                if exists(syzygy + f + '/' + wdl):
                    print('Exists!')
                    # hs = open(syzygy_endgame + fen_position + '/' + 'checksum.md5', "a")
                    # hs.write(md5(syzygy_endgame + fen_position + '/' + get_wdl) + '  ' + get_wdl + "\n")
                    # hs.close()
                    continue
                _ = wget.download(domain + f + '/' + wdl,
                                  syzygy + f + '/' + wdl, bar=bar_progress)


if __name__ == "__main__":
    main()
