from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np

# Define a few functions to collect data as the model runs
def proportion_susceptible(model):
    states = np.array([agent.state for agent in model.schedule.agents])
    susceptible = sum(states==0)/len(states)
    return susceptible

def proportion_infected(model):
    states = np.array([agent.state for agent in model.schedule.agents])
    infected = sum(states==1)/len(states)
    return infected

def proportion_removed(model):
    states = np.array([agent.state for agent in model.schedule.agents])
    removed = sum(states==2)/len(states)
    return removed


class SIRModel(Model):
    """A model with some number of agents."""
    def __init__(self, num_agents, initial_infected, removed_probability, width, height):
        self.num_agents = num_agents
        self.initial_infected = initial_infected
        self.removed_probability = removed_probability
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = SIRAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            
            self.datacollector = DataCollector(
            model_reporters={"susceptible": proportion_susceptible,
                             "infected": proportion_infected,
                             "removed": proportion_removed})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

class SIRAgent(Agent):
    """ An agent in the model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = np.random.choice([0,1], p=[1-self.model.initial_infected,self.model.initial_infected])

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True,
                                                          include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_disease(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            if other.state == 0:  # If susceptible
                other.state = 1
                
    def removed(self):
        """Removed corersponds to recovered or died."""
        p = self.model.removed_probability
        state = np.random.choice([2,1], p=[p, 1-p])
        self.state = state

    def step(self):
        """Take a step forwards in time."""
        self.move()
        if self.state == 1:
            self.give_disease()
            self.removed()

