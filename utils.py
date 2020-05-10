import os


def getTerminalSize():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct
            cr = struct.unpack(
                'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except Exception:
            return
        return cr

    try:
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        assert(cr == True)
    except AssertionError:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except Exception:
            cr = (os.environ.get('LINES', 25), os.environ.get('COLUMNS', 80))
    try:
        return int(cr[1]), int(cr[0])
    except Exception:
        raise RuntimeError("could not determine current terminal size")
