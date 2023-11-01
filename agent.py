import random
from model import load_model, identify
from client.state import ClientGameRoundState, ClientGameState
import time
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
AGENT_DIR = os.path.join(ROOT_DIR, "agent")

class PokerAgent(object):

    def __init__(self):
        self._model = load_model()
        self._estimated_card = None

    def make_action(self, state: ClientGameState, round: ClientGameRoundState) -> str:
        """
        Next action, used to choose a new action depending on the current state of the game. This method implements your
        unique PokerBot strategy. Use the state and round arguments to decide your next best move.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game (a game has multiple rounds)
        round : ClientGameRoundState
            State object of the current round (from deal to showdown)

        Returns
        -------
        str in ['BET', 'CALL', 'CHECK', 'FOLD'] (and in round.get_available_actions())
            A string representation of the next action an agent wants to do next, should be from a list of available actions
        """
        # Card Jack
        class Jack(object):

            def choice(self, chance, alpha):
                turn_order = round.get_turn_order()
                moves_history = round.get_moves_history()
                
                # second to play
                if turn_order == 2 and moves_history[-1] == "CHECK":
                    return 'BET' if chance <= 1/3 else "CHECK"
                
                # first to play in the first round
                elif not moves_history:
                    return 'BET' if chance <= alpha else 'CHECK'
                
                # other actions
                return 'FOLD'
            
        # Card King
        class King(object):
            
            def choice(self, chance, alpha):
                turn_order = round.get_turn_order()
                moves_history = round.get_moves_history()
                
                # second to play
                if turn_order == 2 and moves_history[-1] == "CHECK":
                    return 'BET'
                
                # first to play in the first round
                elif not moves_history:
                    return 'BET' if chance <= 3 * alpha else 'CHECK'

                # other actions
                return 'CALL'
        
        # Card Queen
        class Queen(object):
            
            def choice(self, chance, alpha):
                turn_order = round.get_turn_order()
                moves_history = round.get_moves_history()
                
                # second to play
                if turn_order == 2:
                    return "CHECK" if  moves_history[-1] == "CHECK" else "CALL"

                # first to play in the first round
                elif not moves_history:
                    return "CHECK"
                
                # other actions
                return "CALL" if chance <= (alpha + 1/3) else "FOLD"

        # Wait for response
        time.sleep(0.1)
        
        # Set random variables
        chance = random.uniform(0, 1) 
        alpha = random.uniform(0, (1/3))

        current_card = round.get_card()[0]

        # Perform image recognition if image is not yet known
        if current_card == '?':
            current_card = self.get_estimated_card()[0]
            
        if current_card not in ['J', 'Q', 'K']:
            self.on_error("Inavailable Input!") 

        # Dictionary to map cards to correct objects
        card_mapper = {"J": Jack(),"K": King(),"Q": Queen()}
        card_object = card_mapper[current_card]

        # Return the choice
        choice = card_object.choice(chance,alpha)
        return choice
        
    def on_image(self, image):
        """
        This method is called every time when the card image changes. Use this method for image recongition.

        Parameters
        ----------
        image : Image
            Image object
        """
        try:
            self.set_estimated_card(identify(image,self._model))
            if self.get_estimated_card == None:
                print("Identification failed. Set card as J.")
                self.set_estimated_card('J')

        except Exception:
            self.on_error("on image error")
    
    def set_estimated_card(self, card:str):
        """
        Setter for attribute _estimated_card.
        """
        self._estimated_card = card

    
    def get_estimated_card(self):
        """
        Getter for attribute estimated_card
        """
        return self._estimated_card


    def on_error(self, error):
        """
        This methods will be called in case of error either from the server backend or from the client itself.
        You can also use this function for error handling.

        Parameters
        ----------
        error : str
            string representation of the error
        """

        print(error)
        raise Exception(error)

    def on_game_start(self):
        """
        This method will be called once at the beginning of the game when the server has confirmed that both players are connected.
        """
        # print('game has started')
        pass

    def on_new_round_request(self, state: ClientGameState):
        """
        This method is called every time before a new round is started. A new round is started automatically.
        You can use this method for logging purposes.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        """
        try:
            logFile = open(AGENT_DIR, 'a')
            logFile.write('\n New round ...\n')
            logFile.write('Coordinator ID: '+ state.get_coordinator_id()+'.\n')
            logFile.write('Player token: '+ state.get_player_token()+'.\n')
            logFile.close()
        except Exception:
            self.on_error("On_new_round_request error.")

    def on_round_end(self, state: ClientGameState, round: ClientGameRoundState):
        """
        This method is called every time a round has ended. A round ends automatically. 
        You can use this method for logging purposes.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        round : ClientGameRoundState
            State object of the current round
        """
        try:
            logFile = open(AGENT_DIR, 'a')
            logFile.write(f'----- Round { round.get_round_id() } results ----- ')
            logFile.write(f'  Your card       : { round.get_card() }')
            logFile.write(f'  Your turn order : { round.get_turn_order() }')
            logFile.write(f'  Moves history   : { round.get_moves_history() }')
            logFile.write(f'  Your outcome    : { round.get_outcome() }')
            logFile.write(f'  Current bank    : { state.get_player_bank() }')
            logFile.write(f'  Show-down       : { round.get_cards() }')
            logFile.close()

        except Exception:
            self.on_error('on_round_end error.')

    def on_game_end(self, state: ClientGameState, result: str):
        """
        This method is called once after the game has ended. A game ends automatically. 
        You can use this method for logging purposes.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        result : str in ['WIN', 'DEFEAT']
            End result of the game
        """
        try:
            logFile = open(AGENT_DIR, 'a')
            current_bank = state.get_player_bank()
            logFile.write(f'\n Game end...\nResult: {result}.\nCurrent bank:{current_bank}.\n')
            print(f'Result: {result}')
            logFile.close()

        except Exception:
            self.on_error('On_game_end error.')
      
