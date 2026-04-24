from src.muzero.temperature import temperature_for_step


def test_temperature_segments():
    schedule = [(0, 1.0), (100, 0.5), (200, 0.25)]
    assert temperature_for_step(0, schedule) == 1.0
    assert temperature_for_step(50, schedule) == 1.0
    assert temperature_for_step(100, schedule) == 0.5
    assert temperature_for_step(199, schedule) == 0.5
    assert temperature_for_step(200, schedule) == 0.25
    assert temperature_for_step(10_000, schedule) == 0.25
