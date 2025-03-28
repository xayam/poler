
from p.uci import UCI


def main() -> None:
    uciLoop = UCI()
    while True:
        command = input()
        uciLoop.process_command(command)
        if command == "quit":
            break


if __name__ == "__main__":
    main()
