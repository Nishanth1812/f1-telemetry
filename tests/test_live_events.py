from f1telemetry.live_car import DrivenCarSource
from f1telemetry.synthetic import Frame, FrameEvent


def test_saved_events_follow_the_shifted_live_timeline() -> None:
    source = DrivenCarSource({"speed": 0.0})
    event = FrameEvent("fault", 0, fault_type="spike", channel="speed", label=True)
    source.start_run((Frame(0, {"speed": 1.0}, (event,)),))
    deliveries = tuple(source.iter_frames(100_000))
    events = [item for frame in deliveries for item in frame.events]
    assert len(events) == 1
    assert events[0].time_us == 1
    assert events[0].fault_type == "spike"
    for delivery in deliveries:
        assert all(not sample.events for sample in delivery.samples)
