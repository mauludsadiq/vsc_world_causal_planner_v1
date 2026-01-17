from text_world.agent.action_parser import predict_action9_topk
from text_world.actions import ALL_ACTIONS

def test_action_parser_never_out_of_range():
    texts = [
        "open the door",
        "look around",
        "exit",
        "help",
        "take the key",
        "close the door",
        "use the key",
        "talk to them",
        "random unknown utterance xyz",
    ]
    for t in texts:
        out = predict_action9_topk(t, k=9, seed=0)
        assert len(out.action_ids) >= 1
        for a in out.action_ids:
            assert a in ALL_ACTIONS
