from mesa import Agent

class Shooter(Agent):
	#superclass agent init
	def __init(self, unique_id, model):
		super().__init__(unique_id, model)
		self.health = 100
		self.bullets = []
		self.cooldown = 0

	#behaviours
	def turnLeft(self):
		pass


	def turnRight(self):
		pass


	def moveForward(self):
		pass


	def shoot(self):
		pass


"""agent types"""
#random testing agent
class RandomShooter(Shooter):
	def __init__(self, unique_id, model):
		Shooter.__init__(self, unique_id, model)


	def step(self):
		pass


#learning agent with keras
class ReinforcementLearningShooter(Shooter):
	def __init__(self, unique_id, model):
		Shooter.__init__(self, unique_id, model)


	def step(self):
		pass	



