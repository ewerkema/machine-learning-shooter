from mesa import Model
from agent import RandomShooter, ReinforcementLearningShooter

class World(Model):
	#model setup
	def __init__(self):
		self.drawModel = False
		
		#amount and types of agents
		self.num_agents = 2
		self.num_rl_agents = 1
		self.rl_agents = False

		for i in range(self.num_agents):
			if(rl_agents and i < num_rl_agents): 
				a = ReinforcementLearningShooter(i, self)

			elif(rng_agents):
				a = RandomShooter(i, self)
			self.schedule.add(a)


	#model transformations
	def drawGame(self):
		pass



	#model step
	def step(self):
		#agents perform actions
		self.schedule.step()

		#draw visual representation of the game
		if(self.drawModel):
			self.drawGame()