import client.state


TEST_PLAYER_TOKEN = 'Test_TOKEN'
TEST_COORDINATOR_ID = 'Test_COORDINATOR'
TEST_PLAYER_BANK = 1

test_client = client.state.ClientGameState(TEST_COORDINATOR_ID,TEST_PLAYER_TOKEN,TEST_PLAYER_BANK)


class TestClientGameState:
    def test_client_game_state_get_methods(self):
        # Test for the correct set and get methods
        assert test_client.get_player_token() == TEST_PLAYER_TOKEN
        assert test_client.get_coordinator_id() == TEST_COORDINATOR_ID
        assert test_client.get_player_bank() == TEST_PLAYER_BANK
        
    
    def test_client_game_state_start_new_round(self):
        # Test for the correct type of new round
        test_client.start_new_round()
        assert type(test_client.get_rounds()).__name__ == 'list'
        assert test_client.get_last_round_state().__class__.__name__ == 'ClientGameRoundState'

    def test_client_game_state_update_bank(self):
        # Test for the correct update bank method
        test_client.update_bank('2')
        assert test_client.get_player_bank() == TEST_PLAYER_BANK + 2