"""Verification wrapper: run an entry point with socket networking disabled."""
from pathlib import Path
import runpy
import socket
import sys


def deny_network(*args, **kwargs):
    raise RuntimeError('Network access is disabled for this execution check')


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -I offline_entrypoint.py SCRIPT [ARGS ...]')
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.socket.sendto = deny_network
    try:
        socket.create_connection(('127.0.0.1', 9))
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Network guard did not install')
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
