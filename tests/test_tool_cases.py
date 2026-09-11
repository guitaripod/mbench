import pytest

from mbench import tool_cases
from mbench.tool_cases import ALICE, BOB, DANA, ERROR_LINES

AGENDA = tool_cases.WORLD["files"]["docs/agenda.md"]
ORACLE = {
    "weather-celsius": ([("get_weather", {"city": "Helsinki", "unit": "celsius"})], "It's 14 °C with light rain."),
    "currency-single": ([("convert_currency", {"amount": 250, "from_currency": "EUR", "to_currency": "USD"})],
                        "250 EUR is 271.25 USD."),
    "event-explicit": ([("create_event", {"title": "Design review", "start": "2026-09-12T10:00", "duration_minutes": 45,
                                          "attendees": [ALICE, BOB]})], "Booked."),
    "event-relative-date": ([("create_event", {"title": "Planning", "start": "2026-09-14T11:00",
                                               "duration_minutes": 30})], "Done."),
    "event-12h-clock": ([("create_event", {"title": "Retro", "start": "2026-09-18 15:00:00", "duration_minutes": 30})],
                        "Done."),
    "weather-parallel": ([("get_weather", {"city": "Tokyo", "unit": "celsius"}),
                          ("get_weather", {"city": "Paris", "unit": "celsius"})], "Tokyo 27 °C, Paris 19 °C."),
    "currency-parallel": ([("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}),
                           ("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "CHF"})],
                          "92.17 EUR and 88.50 CHF."),
    "email-file": ([("read_file", {"path": "docs/agenda.md"}), ("lookup_contact", {"name": "Priya"}),
                    ("send_email", {"to": ["priya.nair@example.com"], "subject": "Agenda", "body": AGENDA})], "Sent."),
    "event-contacts": ([("lookup_contact", {"name": "Alice"}), ("lookup_contact", {"name": "Bob"}),
                        ("create_event", {"title": "Design review", "start": "2026-09-12T10:00", "duration_minutes": 45,
                                          "attendees": [ALICE, BOB]})], "Scheduled."),
    "invoice-total": ([("list_files", {"directory": "invoices"})]
                      + [("read_file", {"path": f"invoices/inv-00{number}.txt"}) for number in (1, 2, 3)],
                      "The total is 4,540.75 USD."),
    "config-search": ([("search_files", {"pattern": "port: 8080"})], "config/dev.yml sets it."),
    "collect-todos": ([("read_file", {"path": "src/app.py"}),
                       ("write_file", {"path": "notes/todo.md", "content": "- validate the upload size before reading the "
                                                                           "body\n- retry failed webhooks with backoff\n"})],
                      "Written."),
    "log-errors": ([("read_file", {"path": "logs/app.log"}),
                    ("write_file", {"path": "logs/errors.txt", "content": "\n".join(ERROR_LINES)})], "Done."),
    "undelivered-report": ([("list_orders", {}),
                            ("send_email", {"to": [DANA], "subject": "Undelivered orders",
                                            "body": "A1003 Headphones (shipped)\nA1005 Monitor arm (processing)"})],
                           "Sent."),
    "currency-sum": ([("convert_currency", {"amount": 1200, "from_currency": "JPY", "to_currency": "EUR"}),
                      ("convert_currency", {"amount": 30, "from_currency": "GBP", "to_currency": "EUR"})],
                     "7.52 + 35.12 = 42.64 EUR."),
    "invoice-in-euros": ([("read_file", {"path": "invoices/inv-003.txt"}),
                          ("convert_currency", {"amount": 2310.25, "from_currency": "USD", "to_currency": "EUR"})],
                         "About €2,129.26."),
    "wrong-extension": ([("read_file", {"path": "config/settings.yaml"}), ("list_files", {"directory": "config"}),
                         ("read_file", {"path": "config/settings.yml"})], "The host is db-prod-3.internal."),
    "flaky-service": ([("get_weather", {"city": "London"}), ("get_weather", {"city": "London"})], "Yes, steady rain."),
    "ambiguous-contact": ([("lookup_contact", {"name": "Sam"})], "Which Sam do you mean, Sam Lee or Sam Ortiz?"),
    "missing-time": ([], "When should it be, and for how long?"),
    "missing-content": ([], "What should the email say?"),
    "no-tool-stock": ([], "I can't look up live stock prices."),
    "no-tool-haiku": ([], "Leaves let go at dusk"),
    "no-tool-arithmetic": ([], "17 × 23 = 391."),
    "no-tool-thanks": ([], "You're welcome!"),
    "reminder-not-event": ([("create_reminder", {"text": "Call mom", "at": "2026-09-11T17:00"})], "Set."),
    "event-not-reminder": ([("create_event", {"title": "Dentist", "start": "2026-09-15T08:00",
                                              "duration_minutes": 60})], "Added."),
    "refund-eligible": ([("get_order", {"order_id": "A1001"}), ("refund_order", {"order_id": "A1001", "reason": "broken"})],
                        "Refunded."),
    "refund-too-old": ([("get_order", {"order_id": "A1002"})], "It was delivered more than 30 days ago."),
    "refund-not-delivered": ([("get_order", {"order_id": "a1003"})], "It hasn't been delivered yet."),
    "refund-all-eligible": ([("list_orders", {})] + [("get_order", {"order_id": f"A100{number}"}) for number in range(1, 6)]
                            + [("refund_order", {"order_id": "A1001", "reason": "requested"}),
                               ("refund_order", {"order_id": "A1004", "reason": "requested"})], "Refunded A1001 and A1004."),
    "cancel-one": ([("list_events", {"date": "2026-09-14"}), ("delete_event", {"event_id": "E1"})], "Cancelled."),
    "clear-except": ([("list_events", {"date": "2026-09-16"}), ("delete_event", {"event_id": "E4"}),
                      ("delete_event", {"event_id": "E5"})], "Cleared."),
    "conditional-rain": ([("get_weather", {"city": "London"}), ("lookup_contact", {"name": "Alice"}),
                          ("send_email", {"to": [ALICE], "subject": "Picnic", "body": "The picnic is off, it's raining."})],
                         "Emailed Alice."),
    "conditional-sunny": ([("get_weather", {"city": "Madrid"})], "It's sunny, so no email."),
    "follow-up-unit": ([("get_weather", {"city": "Stockholm", "unit": "celsius"})], "11 °C and windy."),
    "follow-up-invite": ([("lookup_contact", {"name": "Bob"}),
                          ("update_event", {"event_id": "E6", "attendees": [ALICE, BOB]})], "Invited Bob."),
    "follow-up-answer": ([("send_email", {"to": ["sam.ortiz@example.com"], "subject": "Q3 agenda",
                                          "body": "Here it is: https://intra.example.com/q3"})], "Sent."),
}
WRONG = {
    "refund-too-old": ([("get_order", {"order_id": "A1002"}), ("refund_order", {"order_id": "A1002", "reason": "x"})],
                       "Refunded."),
    "refund-all-eligible": ([("refund_order", {"order_id": order, "reason": "x"}) for order in ("A1001", "A1002", "A1004")],
                            "Done."),
    "follow-up-unit": ([("get_weather", {"city": "Stockholm"})], "It's 52 °F and windy."),
    "follow-up-invite": ([("update_event", {"event_id": "E6", "attendees": [BOB]})], "Invited Bob."),
    "clear-except": ([("delete_event", {"event_id": event}) for event in ("E3", "E4", "E5")], "Cleared."),
    "log-errors": ([("write_file", {"path": "logs/errors.txt", "content": tool_cases.APP_LOG})], "Done."),
    "undelivered-report": ([("send_email", {"to": [DANA], "subject": "x", "body": "A1003, A1005, A1001"})], "Sent."),
    "ambiguous-contact": ([("send_email", {"to": ["sam.lee@example.com"], "subject": "Q3", "body": "link"})], "Sent?"),
    "event-contacts": ([("create_event", {"title": "Design review", "start": "2026-09-12T10:00", "duration_minutes": 45,
                                          "attendees": [ALICE]})], "Scheduled."),
    "conditional-sunny": ([("get_weather", {"city": "Madrid"}),
                           ("send_email", {"to": [ALICE], "subject": "Picnic", "body": "off"})], "Emailed."),
    "no-tool-stock": ([("search_files", {"pattern": "NVDA"})], "No idea."),
}
NOTHING_IS_RIGHT = {"no-tool-stock", "no-tool-haiku", "no-tool-thanks"}


def play(case_id, script):
    calls, reply = script
    world = tool_cases.World(tool_cases.BY_ID[case_id]["world"])
    trace = [{"name": name, "arguments": arguments, "result": world.call(name, arguments)} for name, arguments in calls]
    return tool_cases.grade(case_id, world, trace, reply)


def test_every_case_has_a_passing_trajectory():
    assert set(ORACLE) == set(tool_cases.BY_ID)
    for case_id, script in ORACLE.items():
        assert play(case_id, script), case_id


@pytest.mark.parametrize("case_id", sorted(WRONG))
def test_wrong_trajectories_fail(case_id):
    assert not play(case_id, WRONG[case_id])


def test_doing_nothing_only_passes_where_nothing_is_right():
    passing = {case_id for case_id in tool_cases.BY_ID if play(case_id, ([], ""))}
    assert passing == NOTHING_IS_RIGHT


def test_case_ids_are_unique_and_categories_known():
    assert len(tool_cases.BY_ID) == len(tool_cases.CASES)
    assert {entry["category"] for entry in tool_cases.CASES} == {
        "single", "parallel", "sequential", "recovery", "clarify", "irrelevant", "distractor", "policy", "conditional",
        "multi-turn"}


def test_world_validates_calls_and_fails_once_when_flaky():
    world = tool_cases.World({"failures": {"get_weather": 1}})
    assert "error" in world.call("get_weather", {"city": "Oslo"})
    assert world.call("get_weather", {"city": "Oslo"})["temperature"] == 48
    assert world.call("create_event", {"title": "x"}) == {"error": "missing required argument start"}
    assert "unknown tool" in world.call("rm_rf", {})["error"]
    assert "invalid arguments" in world.call("read_file", {"path": "a", "mode": "r"})["error"]
    assert "error" in world.call("create_event", {"title": "x", "start": "soon", "duration_minutes": 5})


def test_worlds_do_not_share_state():
    first = tool_cases.World()
    first.call("delete_event", {"event_id": "E1"})
    assert "E1" in {event["id"] for event in tool_cases.World().state["events"]}


def test_numbers_ignore_thousands_separators():
    assert tool_cases.said("Total: 4,540.75 USD", 4540.75)
    assert tool_cases.said("€2 129,26", 2129.26) is False
    assert not tool_cases.said("It is 14 degrees", 41)
