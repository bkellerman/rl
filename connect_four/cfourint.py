import sys
import cfour
import cfouragent

COLS = 7
ROWS = 6
WIN = 4
if __name__ == '__main__':
    model_name = sys.argv[1]
    for _ in range(5):
        print("New game")
        game = cfour.Game(COLS, ROWS, WIN)
        agent = cfouragent.DQNConnectFourAgent(COLS, ROWS, WIN)
        agent.load_game(model_name)

        while True:
            game.print_board()

            # User input
            while True:
                try:
                    col = int(input("Red's turn: "))
                    inserted = game.insert(col, 'R')
                except Exception:
                    print("Invalid input. Try again.")
                    continue
                if not inserted:
                    print("Invalid move. Try again.")
                    continue
                break

            if game.check_for_win():
                print("You win!")
                game.print_board()
                break

            # Agent input
            state = game.state()
            agent_col = agent.choose_game_action(state)
            game.insert(agent_col, 'Y')

            if game.check_for_win():
                print("Agent wins!")
                game.print_board()
                break

            if game.full():
                print("Tie!")
                game.print_board()
                break
