import h.model.barriers.chess.uci as uci


def main() -> None:
    uciLoop = uci.UCI()

    while True:
        command = input()
        uciLoop.process_command(command)

        if command == "quit":
            break


if __name__ == "__main__":
    main()
