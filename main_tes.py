


def myy_print(sackl):
    print(sackl)

def aundb(a, b):
    c = a+b
    print(c)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--printer', metavar='path', required=True,
                        help='the path to workspace')
    parser.add_argument('--a', type=int, choices=range(1, 10), required=False,
                        help='the path to workspace')
    parser.add_argument('--b', type=int, required=False,
                        help='the path to workspace')
    args = parser.parse_args()
    myy_print(args.printer)
    aundb(args.a, args.b)