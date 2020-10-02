# Train the Game

This project demonstrates an idea of training the neural network to play the game autonomously by observing player's actions.

## The game

The game itself is just a car riding the road. There are possible turns and obstacles on the way. The NN should learn how to steer in order to follow the road and avoid collisions.

## NN data

### Input

- a patch of the road in front of the car
- a car's distance from the right edge of the road
- a car's angle relative to the North
- user's angle


### Output

A new angle of the car.


## Technologies

- PyTorch
- PyGame