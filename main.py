import sys
import config
from world import World


def main():
    world = World()

    for epoch in range(config.epochs):
        running = world.run_epoch(epoch)

        if not running:
            break

    print("Total wins per player:")
    print(world.players_won)

    world.save_results_to_excel()
    world.save_models()
    world.quit()


if __name__ == '__main__':
    sys.exit(main())
