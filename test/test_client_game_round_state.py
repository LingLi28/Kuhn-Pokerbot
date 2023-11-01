import os
import pytest
from PIL import Image
from client.state import ClientGameRoundState

TEST_COORDINATOR_ID = "0000" 
TEST_ROUND_ID = 1
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

round_state = ClientGameRoundState(TEST_COORDINATOR_ID, TEST_ROUND_ID)


class TestClientGameRoundState:
    def test_get_coordinator_id(self):
        assert round_state.get_coordinator_id() == TEST_COORDINATOR_ID
    
    def test_get_round_id(self):
        assert round_state.get_round_id() == TEST_ROUND_ID
    
    def test_card(self):     
        # Test for correct get and set behavior
        round_state.set_card("J")
        assert round_state.get_card() == "J"
        
    def test_card_image(self):
        path = os.path.join(TEST_DIR, "data_sets/test_images/J_1.png")
        img = Image.open(path)
        # Test for correct get and set behavior
        round_state.set_card_image(img)
        assert round_state.get_card_image() == img
        
    def test_turn_order(self):
        # Test for correct get and set behavior
        round_state.set_turn_order(int(1))
        assert round_state.get_turn_order() == int(1)       
        
    def test_move_history(self):
        # Test for correct get and set behavior
        round_state.set_moves_history(["BET"])
        assert round_state.get_moves_history() == ["BET"]
        
        # Test if add move does add a move to the move history
        round_state.add_move_history("FOLD")
        assert round_state.get_moves_history() == ["BET", "FOLD"]
        
    def test_available_actions(self):
        # Test for correct get and set behavior
        round_state.set_available_actions(['BET', 'CHECK', 'FOLD'])
        assert round_state.get_available_actions() == ['BET', 'CHECK', 'FOLD']
    
    def test_outcome(self):
        # Test for correct get and set behavior
        round_state.set_outcome("1")
        assert round_state.get_outcome() == "1"
    
    def test_cards(self):
        # Test for correct get and set behavior
        round_state.set_cards("KJ")
        assert round_state.get_cards() == "KJ"
        
        # Test if unknown card does work as well
        round_state.set_cards("K?")
        assert round_state.get_cards() == "K?"
        
        