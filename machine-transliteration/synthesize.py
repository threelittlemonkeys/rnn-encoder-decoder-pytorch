import sys
import random

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

minlen = 5
maxlen = 20
data_size = int(sys.argv[1])

def transliterate(x):

    if "A" <= x <= "Z":
        return chr(ord(x) + 32)

    if "a" <= x <= "z":
        return chr(ord(x) - 32)

    return x

for _ in range(data_size):

    xs = [random.choice(chars) for _ in range(random.randint(minlen, maxlen))]
    ys = [transliterate(x) for x in xs]

    print(" ".join(xs), " ".join(ys), sep = "\t")
