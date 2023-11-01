import pytest
import os
from agent import *
from client.state import ClientGameRoundState, ClientGameState

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
AGENT_DIR = os.path.join(ROOT_DIR, "agent")


TEST_PLAYER_TOKEN = 'Test_TOKEN'
TEST_COORDINATOR_ID = 'Test_COORDINATOR'
TEST_PLAYER_BANK = 1


TEST_AGENT = PokerAgent()
TEST_STATE = ClientGameState(TEST_COORDINATOR_ID, TEST_PLAYER_TOKEN, TEST_PLAYER_BANK)
TEST_ROUND_1 = ClientGameRoundState(TEST_COORDINATOR_ID, '1')
TEST_ROUND_2 = ClientGameRoundState(TEST_COORDINATOR_ID, '2')


class TestPokerAgent:
    def test_init(self):
        # Test for the correct initial card
        assert TEST_AGENT.get_estimated_card() == None
    
    def test_set_estimated_card(self):
        # Test for the correct estimated card
        TEST_AGENT.set_estimated_card('J')
        assert TEST_AGENT.get_estimated_card() == 'J'

    def test_make_action_jack(self):
        # Test first round behavior 
        TEST_AGENT.on_game_start()
        TEST_ROUND_1.set_turn_order(1)
        TEST_ROUND_1.set_card(['J'])
        TEST_ROUND_1.set_moves_history([])
        TEST_ROUND_1.set_available_actions(['BET', 'CHECK', 'FOLD'])
        # Test for the correct return of make_action in the available actions list
        assert TEST_AGENT.make_action(TEST_STATE, TEST_ROUND_1) in ['CHECK','BET']
        
        # Test third round behavior 
        TEST_ROUND_1.set_moves_history(['CHECK', 'BET'])
        # Test for the correct return of make_action in the available actions list
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_1) in ['FOLD','CALL']

        # Test second round if first round was 'CHECK'
        TEST_ROUND_2.set_turn_order(2)
        TEST_ROUND_2.set_moves_history(['CHECK'])
        TEST_ROUND_2.set_card(['J'])
        TEST_ROUND_2.set_available_actions(['CHECK','BET'])
        # Test for the correct return of make_action in the available actions list
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_2) in ['CHECK','BET']

        # Test second round if first round was 'BET'
        TEST_ROUND_2.set_moves_history(['BET'])
        TEST_ROUND_2.set_card(['J'])
        TEST_ROUND_2.set_available_actions(['FOLD','CALL'])
        #Test for the correct return of make_action in the available actions list
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_2) in ['FOLD','CALL']

    def test_make_action_queen(self):
        # Test first round behavior 
        TEST_AGENT.on_game_start()
        TEST_ROUND_1.set_turn_order(1)
        TEST_ROUND_1.set_card(['Q'])
        TEST_ROUND_1.set_moves_history([])
        TEST_ROUND_1.set_available_actions(['BET', 'CHECK', 'FOLD'])
        assert TEST_AGENT.make_action(TEST_STATE, TEST_ROUND_1) in ['CHECK','BET']

        # Test third round behavior 
        TEST_ROUND_1.set_moves_history(['CHECK', 'BET'])
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_1) in ['FOLD','CALL']

        # Test second round if first round was 'CHECK'
        TEST_ROUND_2.set_turn_order(2)
        TEST_ROUND_2.set_moves_history(['CHECK'])
        TEST_ROUND_2.set_card(['Q'])
        TEST_ROUND_2.set_available_actions(['CHECK','BET'])
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_2) in ['CHECK','BET']

        # Test second round if first round was 'BET'
        TEST_ROUND_2.set_moves_history(['BET'])
        TEST_ROUND_2.set_card(['Q'])
        TEST_ROUND_2.set_available_actions(['FOLD','CALL'])
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_2) in ['FOLD','CALL']


    def test_make_action_king(self):
        # Test first round behavior 
        TEST_AGENT.on_game_start()
        TEST_ROUND_1.set_turn_order(1)
        TEST_ROUND_1.set_card(['K'])
        TEST_ROUND_1.set_moves_history([])
        TEST_ROUND_1.set_available_actions(['BET', 'CHECK', 'FOLD'])
        assert TEST_AGENT.make_action(TEST_STATE, TEST_ROUND_1) in ['CHECK','BET']

        # Test third round behavior 
        TEST_ROUND_1.set_moves_history(['CHECK', 'BET'])
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_1) in ['FOLD','CALL']

        # Test second round if first round was 'CHECK'
        TEST_ROUND_2.set_turn_order(2)
        TEST_ROUND_2.set_moves_history(['CHECK'])
        TEST_ROUND_2.set_card(['K'])
        TEST_ROUND_2.set_available_actions(['CHECK','BET'])
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_2) in ['CHECK','BET']

        # Test second round if first round was 'BET'
        TEST_ROUND_2.set_moves_history(['BET'])
        TEST_ROUND_2.set_card(['K'])
        TEST_ROUND_2.set_available_actions(['FOLD','CALL'])
        assert TEST_AGENT.make_action(TEST_STATE,TEST_ROUND_2) in ['FOLD','CALL']



    def test_error_handeling(self):
        """
        Test on_error()
        """
        assert pytest.raises(Exception, TEST_AGENT.on_error,'TestError')