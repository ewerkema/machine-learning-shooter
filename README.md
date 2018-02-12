# Shooter Game with Feed Forward and Recurrent Neural Networks

The game can be run by running the main script in ```main.py```. The configuration of players can be done in the ```config.py``` using the following variables:

| Variable              | Default | Description  |
| --------------------- |-------------| -----|
| total_players         | 2     | Total number of players |
| epochs                | 5     | Total number of games |
| fps                   | 25    | Amount of frames per second (game length is 20s) |
| players               |       | List of player configurable variables |
| players.feed_forward  | True  | Boolean for enabling Feed Forward or LSTM neural networks |
| players.random        | False | Boolean for letting the player behave randomly |
| players.hidden_size   | 50    | Amount of hidden neurons |