# Cronus
Cronus is a Rocket League bot trained via reinforcement learning. It utilizes [RLGym](https://rlgym.org/) (A python API to treat the game Rocket League as an OpenAI Gym environment) to run many concurrent instances of Rocket League at 100x speed. It has played over 3 years of game time to reach its current level and is continually training to gain more experience.

## What make Cronus different than other Rocket League bots?
Cronus draws inspiration from several of the community bots in development in the RLGym community. As a differentiator, it has specific aerial state setters and custom rewards designed to target a style of play with a more advanced mechanical focus - one of the hardest areas for a bot to learn due to the very large observation space and the many precise actions required to reach certain states. Ideally, this bot will learn how to air dribble and double tap, while maintaining a focus on winning the game overall. Additionally, this project explores the maximum learnings the agent can reach with compute constraints and consequently a relatively smaller network size. 

## Understanding the repo
To run training, simply call `python learner.py`. The other files contain supporting classes for the model training, including custom rewards, state setters, terminal conditions, and action parsers. The rlbot-support folder contains the necessary files to run the resulting agent in RLBot and play against other bots and/or humans.

## How to build your own
If you want to build your own rocket league reinforcement learning bot, check out the links listed below. The RLGym discord is a great place to see other ongoing initiatives in the reinforcement learning bot development process.

## Resources
- [RLGym](https://rlgym.org/)
- [RLBot](https://rlbot.org/)
- [RLGym discord](https://discord.com/invite/NjAHcP32Ae)
