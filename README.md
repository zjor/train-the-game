# Train the Game

This project demonstrates an idea of training the neural network to play the game autonomously by observing player's actions.

## The game

The game itself is just a car riding the road. There are possible turns and obstacles on the way. The NN should learn how to steer in order to follow the road and avoid collisions.

![screenshot](https://habrastorage.org/webt/ax/eo/ls/axeolseo46nsnu_poxm6od3ouxs.png)

## NN data

### Input
Lidar readings

### Output
Command: (Left, Right, Straight)

## Frameworks
- PyTorch
- PyGame

## References

- [Habr](https://habr.com/ru/post/526872/)