from text_world.agent.state_renderer import render_state

def test_render_state_contract_narration():
    r = render_state(80)
    assert isinstance(r.narration, str)
    assert "look around" in r.narration.lower()
