import copy
import json
import re
from datetime import datetime

REFUND_WINDOW_DAYS = 30
SYSTEM = (
    "You are an assistant with access to the user's workspace: files, calendar, contacts, email and orders from "
    "the user's shop account. The user is Dana Novak (dana@example.com). Today is Friday 2026-09-11. "
    "Use the tools when a request needs them, and ask the user when a request is missing something only they can tell you. "
    f"Shop policy: an order can be refunded only if it has been delivered and the delivery was at most {REFUND_WINDOW_DAYS} days ago."
)


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
TOOLS = [
    tool("get_weather", "Current weather for a city.", {
        "city": STRING,
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Defaults to fahrenheit."},
    }, ["city"]),
    tool("convert_currency", "Convert an amount between ISO-4217 currencies at today's rate.", {
        "amount": {"type": "number"}, "from_currency": STRING, "to_currency": STRING,
    }, ["amount", "from_currency", "to_currency"]),
    tool("list_files", "List the files under a directory of the workspace.", {"directory": STRING}, ["directory"]),
    tool("read_file", "Read a text file from the workspace.", {"path": STRING}, ["path"]),
    tool("write_file", "Create or overwrite a text file in the workspace.", {"path": STRING, "content": STRING},
         ["path", "content"]),
    tool("search_files", "Search file contents with a regular expression; returns path:line: text for each match.", {
        "pattern": STRING, "directory": {"type": "string", "description": "Defaults to the whole workspace."},
    }, ["pattern"]),
    tool("lookup_contact", "Find contacts whose name contains the given text.", {"name": STRING}, ["name"]),
    tool("create_event", "Create a calendar event.", {
        "title": STRING,
        "start": {"type": "string", "description": "Local date-time, e.g. 2026-09-12T10:00"},
        "duration_minutes": INTEGER,
        "attendees": {"type": "array", "items": STRING, "description": "Email addresses."},
    }, ["title", "start", "duration_minutes"]),
    tool("update_event", "Change fields of an existing calendar event; fields left out stay as they are.", {
        "event_id": STRING, "title": STRING, "start": STRING, "duration_minutes": INTEGER,
        "attendees": {"type": "array", "items": STRING, "description": "The full new list of email addresses."},
    }, ["event_id"]),
    tool("list_events", "List the calendar events on one day.", {"date": {"type": "string", "description": "YYYY-MM-DD"}},
         ["date"]),
    tool("delete_event", "Delete a calendar event.", {"event_id": STRING}, ["event_id"]),
    tool("create_reminder", "Set a reminder that notifies the user at a time. Not a calendar event.", {
        "text": STRING, "at": {"type": "string", "description": "Local date-time, e.g. 2026-09-11T17:00"},
    }, ["text", "at"]),
    tool("send_email", "Send an email from the user's account.", {
        "to": {"type": "array", "items": STRING, "description": "Email addresses."}, "subject": STRING, "body": STRING,
    }, ["to", "subject", "body"]),
    tool("list_orders", "List the user's shop orders with their status.", {}, []),
    tool("get_order", "Details of one shop order, including its delivery date.", {"order_id": STRING}, ["order_id"]),
    tool("refund_order", "Refund a shop order.", {"order_id": STRING, "reason": STRING}, ["order_id", "reason"]),
]
REQUIRED = {entry["function"]["name"]: entry["function"]["parameters"]["required"] for entry in TOOLS}
MUTATING = {"write_file", "create_event", "update_event", "delete_event", "create_reminder", "send_email", "refund_order"}

WEATHER = {
    "helsinki": (14, "light rain"), "tokyo": (27, "humid and clear"), "paris": (19, "cloudy"),
    "london": (12, "steady rain"), "madrid": (31, "sunny"), "oslo": (9, "overcast"),
    "stockholm": (11, "windy"), "berlin": (17, "partly cloudy"),
}
USD_PER_UNIT = {"USD": 1.0, "EUR": 1.085, "GBP": 1.27, "JPY": 0.0068, "CHF": 1.13}
APP_PY = """import os

MAX_UPLOAD = 10 * 1024 * 1024


def upload(request):
    # TODO: validate the upload size before reading the body
    data = request.read()
    return store(data)


def deliver(hook):
    # TODO: retry failed webhooks with backoff
    return hook.send()
"""
APP_LOG = """2026-09-10 08:00:01 INFO server started on :8443
2026-09-10 08:02:13 INFO user 42 logged in
2026-09-10 08:05:47 ERROR payment gateway timeout after 30s (order A1003)
2026-09-10 08:06:02 INFO retrying payment for order A1003
2026-09-10 09:12:30 ERROR disk usage above 90% on /var/lib/app
2026-09-10 10:44:09 INFO nightly export finished
2026-09-10 11:01:55 ERROR webhook to https://hooks.example.com/n1 returned 502
"""
ERROR_LINES = [line for line in APP_LOG.splitlines() if " ERROR " in line]
WORLD = {
    "files": {
        "docs/agenda.md": "# Q3 planning agenda\n1. Hiring plan for the platform team\n2. Budget review: cloud spend is up 18%\n"
                          "3. Launch date for Atlas v2\n",
        "invoices/inv-001.txt": "Invoice INV-001\nClient: Northwind\nAmount: 1250.00 USD\n",
        "invoices/inv-002.txt": "Invoice INV-002\nClient: Contoso\nAmount: 980.50 USD\n",
        "invoices/inv-003.txt": "Invoice INV-003\nClient: Fabrikam\nAmount: 2310.25 USD\n",
        "config/settings.yml": "database:\n  host: db-prod-3.internal\n  port: 5432\nserver:\n  port: 8443\n",
        "config/dev.yml": "server:\n  port: 8080\n  debug: true\n",
        "config/staging.yml": "server:\n  port: 9090\n",
        "src/app.py": APP_PY,
        "logs/app.log": APP_LOG,
    },
    "contacts": [
        {"name": "Alice Chen", "email": "alice.chen@example.com"},
        {"name": "Bob Martin", "email": "bob.martin@example.com"},
        {"name": "Priya Nair", "email": "priya.nair@example.com"},
        {"name": "Sam Lee", "email": "sam.lee@example.com"},
        {"name": "Sam Ortiz", "email": "sam.ortiz@example.com"},
    ],
    "events": [
        {"id": "E1", "title": "Standup", "start": "2026-09-14T09:30", "duration_minutes": 15, "attendees": []},
        {"id": "E2", "title": "Lunch with Priya", "start": "2026-09-14T12:00", "duration_minutes": 60,
         "attendees": ["priya.nair@example.com"]},
        {"id": "E3", "title": "Roadmap review", "start": "2026-09-16T10:00", "duration_minutes": 60,
         "attendees": ["bob.martin@example.com"]},
        {"id": "E4", "title": "1:1", "start": "2026-09-16T14:00", "duration_minutes": 30,
         "attendees": ["alice.chen@example.com"]},
        {"id": "E5", "title": "Vendor call", "start": "2026-09-16T16:00", "duration_minutes": 45, "attendees": []},
    ],
    "orders": {
        "A1001": {"item": "Espresso grinder", "price": 189.0, "status": "delivered", "delivered_on": "2026-09-01"},
        "A1002": {"item": "Desk lamp", "price": 49.0, "status": "delivered", "delivered_on": "2026-06-20"},
        "A1003": {"item": "Headphones", "price": 229.0, "status": "shipped", "delivered_on": None},
        "A1004": {"item": "Keyboard", "price": 119.0, "status": "delivered", "delivered_on": "2026-08-20"},
        "A1005": {"item": "Monitor arm", "price": 89.0, "status": "processing", "delivered_on": None},
    },
    "reminders": [],
    "emails": [],
    "refunds": [],
    "failures": {},
}


def minute(value):
    """A date-time as YYYY-MM-DDTHH:MM, accepting a space separator or seconds; None when it isn't one."""
    try:
        return datetime.fromisoformat(str(value).strip().replace(" ", "T")).strftime("%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def addresses(value):
    items = value if isinstance(value, list) else re.split(r"[,;\s]+", str(value or ""))
    return sorted({str(item).strip().lower() for item in items if str(item).strip()})


class World:
    """The simulated workspace one episode acts on; every tool reads or changes this state and nothing else."""

    def __init__(self, overrides=None):
        self.state = copy.deepcopy(WORLD)
        for key, value in (overrides or {}).items():
            self.state[key] = copy.deepcopy(value)
        self.next_event = 1 + max(int(event["id"][1:]) for event in self.state["events"])

    def call(self, name, arguments):
        handler = getattr(self, f"tool_{name}", None)
        if handler is None or name not in REQUIRED:
            return {"error": f"unknown tool {name}"}
        if not isinstance(arguments, dict):
            return {"error": "arguments must be a JSON object"}
        missing = [key for key in REQUIRED[name] if arguments.get(key) in (None, "")]
        if missing:
            return {"error": f"missing required argument {missing[0]}"}
        if self.state["failures"].get(name):
            self.state["failures"][name] -= 1
            return {"error": "the service timed out; try again"}
        try:
            return handler(**arguments)
        except TypeError as error:
            return {"error": f"invalid arguments: {error}"}
        except (ValueError, KeyError) as error:
            return {"error": str(error)}

    def tool_get_weather(self, city, unit="fahrenheit"):
        key = str(city).split(",")[0].strip().lower()
        if key not in WEATHER:
            return {"error": f"no weather station for {city}"}
        celsius, conditions = WEATHER[key]
        unit = str(unit or "fahrenheit").lower()
        temperature = celsius if unit.startswith("c") else round(celsius * 9 / 5 + 32)
        return {"city": city, "temperature": temperature, "unit": "celsius" if unit.startswith("c") else "fahrenheit",
                "conditions": conditions}

    def tool_convert_currency(self, amount, from_currency, to_currency):
        source, target = str(from_currency).upper(), str(to_currency).upper()
        for code in (source, target):
            if code not in USD_PER_UNIT:
                return {"error": f"unsupported currency {code}"}
        value = round(float(amount) * USD_PER_UNIT[source] / USD_PER_UNIT[target], 2)
        return {"amount": float(amount), "from": source, "to": target, "result": value}

    def tool_list_files(self, directory):
        prefix = str(directory).strip().strip("./").rstrip("/")
        found = sorted(path for path in self.state["files"] if not prefix or path.startswith(prefix + "/"))
        return {"files": found} if found else {"error": f"no such directory: {directory}"}

    def tool_read_file(self, path):
        key = str(path).strip().removeprefix("./")
        if key not in self.state["files"]:
            return {"error": f"no such file: {path}"}
        return {"path": key, "content": self.state["files"][key]}

    def tool_write_file(self, path, content):
        key = str(path).strip().removeprefix("./")
        self.state["files"][key] = str(content)
        return {"ok": True, "path": key, "bytes": len(str(content).encode())}

    def tool_search_files(self, pattern, directory=""):
        prefix = str(directory or "").strip().strip("./").rstrip("/")
        try:
            regex = re.compile(str(pattern))
        except re.error:
            regex = re.compile(re.escape(str(pattern)))
        matches = [f"{path}:{number}: {line}" for path, text in sorted(self.state["files"].items())
                   if not prefix or path.startswith(prefix + "/")
                   for number, line in enumerate(text.splitlines(), 1) if regex.search(line)]
        return {"matches": matches}

    def tool_lookup_contact(self, name):
        needle = str(name).strip().lower()
        return {"contacts": [contact for contact in self.state["contacts"] if needle in contact["name"].lower()]}

    def tool_create_event(self, title, start, duration_minutes, attendees=None):
        if minute(start) is None:
            return {"error": f"start must be a date-time like 2026-09-12T10:00, got {start}"}
        event = {"id": f"E{self.next_event}", "title": str(title), "start": minute(start),
                 "duration_minutes": int(duration_minutes), "attendees": addresses(attendees or [])}
        self.next_event += 1
        self.state["events"].append(event)
        return {"ok": True, "event": event}

    def event(self, event_id):
        for event in self.state["events"]:
            if event["id"] == str(event_id).strip():
                return event
        raise KeyError(f"no event {event_id}")

    def tool_update_event(self, event_id, title=None, start=None, duration_minutes=None, attendees=None):
        event = self.event(event_id)
        if start is not None and minute(start) is None:
            return {"error": f"start must be a date-time like 2026-09-12T10:00, got {start}"}
        changes = {"title": title, "start": minute(start) if start is not None else None,
                   "duration_minutes": int(duration_minutes) if duration_minutes is not None else None,
                   "attendees": addresses(attendees) if attendees is not None else None}
        event.update({key: value for key, value in changes.items() if value is not None})
        return {"ok": True, "event": event}

    def tool_list_events(self, date):
        day = str(date).strip()[:10]
        return {"events": [event for event in self.state["events"] if event["start"].startswith(day)]}

    def tool_delete_event(self, event_id):
        event = self.event(event_id)
        self.state["events"].remove(event)
        return {"ok": True, "deleted": event["id"]}

    def tool_create_reminder(self, text, at):
        if minute(at) is None:
            return {"error": f"at must be a date-time like 2026-09-11T17:00, got {at}"}
        reminder = {"text": str(text), "at": minute(at)}
        self.state["reminders"].append(reminder)
        return {"ok": True, "reminder": reminder}

    def tool_send_email(self, to, subject, body):
        email = {"to": addresses(to), "subject": str(subject), "body": str(body)}
        self.state["emails"].append(email)
        return {"ok": True, "sent_to": email["to"]}

    def tool_list_orders(self):
        return {"orders": [{"order_id": key, "item": order["item"], "status": order["status"]}
                           for key, order in self.state["orders"].items()]}

    def tool_get_order(self, order_id):
        key = str(order_id).strip().upper()
        if key not in self.state["orders"]:
            return {"error": f"no order {order_id}"}
        return {"order_id": key, **self.state["orders"][key]}

    def tool_refund_order(self, order_id, reason):
        key = str(order_id).strip().upper()
        if key not in self.state["orders"]:
            return {"error": f"no order {order_id}"}
        self.state["refunds"].append({"order_id": key, "reason": str(reason)})
        return {"ok": True, "order_id": key, "refunded": self.state["orders"][key]["price"]}


def numbers_in(text):
    cleaned = re.sub(r"(?<=\d)[,\u202f\u00a0](?=\d{3}\b)", "", text or "")
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", cleaned)]


def said(reply, *values):
    found = numbers_in(reply)
    return all(any(abs(number - value) < 0.006 for number in found) for value in values)


def mentions(text, *needles):
    lowered = (text or "").lower()
    return all(needle.lower() in lowered for needle in needles)


def asked(reply):
    return "?" in (reply or "")


def calls_to(calls, name, **expected):
    """The calls to one tool whose arguments contain the expected values, compared case-insensitively."""
    def matches(arguments):
        return all(str(arguments.get(key, "")).strip().lower() == str(value).lower() for key, value in expected.items())

    return [call for call in calls if call["name"] == name and matches(call["arguments"])]


def succeeded(calls, name):
    return [call for call in calls if call["name"] == name and "error" not in call["result"]]


def mutated(calls):
    return [call for call in calls if call["name"] in MUTATING and "error" not in call["result"]]


def title_is(event, title):
    return event["title"].strip().strip("'\"").lower() == title.lower()


def new_events(world):
    original = {event["id"] for event in WORLD["events"]}
    return [event for event in world.state["events"] if event["id"] not in original]


def event_ids(world):
    return {event["id"] for event in world.state["events"]}


def refunded(world):
    return {refund["order_id"] for refund in world.state["refunds"]}


def emails_to(world, address):
    return [email for email in world.state["emails"] if address in email["to"]]


def history_call(call_id, name, arguments, result):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def case(case_id, category, prompt, check, history=(), world=None):
    return {"id": case_id, "category": category, "prompt": prompt, "check": check, "history": list(history),
            "world": world or {}}


ALICE, BOB, DANA = "alice.chen@example.com", "bob.martin@example.com", "dana@example.com"
OSLO_HISTORY = [
    {"role": "user", "content": "What's the weather in Oslo? Give it to me in celsius."},
    *history_call("call_h1", "get_weather", {"city": "Oslo", "unit": "celsius"},
                  {"city": "Oslo", "temperature": 9, "unit": "celsius", "conditions": "overcast"}),
    {"role": "assistant", "content": "It's 9 °C and overcast in Oslo."},
]
BUDGET_HISTORY = [
    {"role": "user", "content": "Create a 30 minute 'Budget sync' with Alice on 2026-09-17 at 13:00."},
    *history_call("call_h1", "lookup_contact", {"name": "Alice"}, {"contacts": [WORLD["contacts"][0]]}),
    *history_call("call_h2", "create_event",
                  {"title": "Budget sync", "start": "2026-09-17T13:00", "duration_minutes": 30, "attendees": [ALICE]},
                  {"ok": True, "event": {"id": "E6", "title": "Budget sync", "start": "2026-09-17T13:00",
                                         "duration_minutes": 30, "attendees": [ALICE]}}),
    {"role": "assistant", "content": "Done: 'Budget sync' with Alice on Thursday 17 September at 13:00 for 30 minutes."},
]
BUDGET_WORLD = {"events": WORLD["events"] + [{"id": "E6", "title": "Budget sync", "start": "2026-09-17T13:00",
                                              "duration_minutes": 30, "attendees": [ALICE]}]}
SAM_HISTORY = [
    {"role": "user", "content": "Send Sam the link to the Q3 agenda."},
    *history_call("call_h1", "lookup_contact", {"name": "Sam"}, {"contacts": WORLD["contacts"][3:5]}),
    {"role": "assistant", "content": "You have two contacts named Sam: Sam Lee and Sam Ortiz. Which one should I email, "
                                     "and what is the link?"},
]


def check_event(world, title, start, duration, attendees=None):
    for event in new_events(world):
        if title_is(event, title) and event["start"] == start and event["duration_minutes"] == duration:
            if attendees is None or set(attendees) <= set(event["attendees"]) <= set(attendees) | {DANA}:
                return True
    return False


CASES = [
    case("weather-celsius", "single", "What's the weather in Helsinki right now? Use celsius.",
         lambda w, calls, reply: calls_to(calls, "get_weather") and said(reply, 14)),
    case("currency-single", "single", "How many US dollars is 250 euros?",
         lambda w, calls, reply: succeeded(calls, "convert_currency") and said(reply, 271.25)),
    case("event-explicit", "single",
         f"Book a 45 minute 'Design review' on 2026-09-12 at 10:00 with {ALICE} and {BOB}.",
         lambda w, calls, reply: check_event(w, "Design review", "2026-09-12T10:00", 45, [ALICE, BOB])),
    case("event-relative-date", "single", "Put a 30 minute 'Planning' on my calendar this coming Monday at 11:00.",
         lambda w, calls, reply: check_event(w, "Planning", "2026-09-14T11:00", 30)),
    case("event-12h-clock", "single", "Schedule 'Retro' on 2026-09-18 at 3pm, half an hour long.",
         lambda w, calls, reply: check_event(w, "Retro", "2026-09-18T15:00", 30)),
    case("weather-parallel", "parallel", "What's the weather in Tokyo and in Paris? Celsius for both.",
         lambda w, calls, reply: len(succeeded(calls, "get_weather")) >= 2 and said(reply, 27, 19)),
    case("currency-parallel", "parallel", "Convert 100 USD to EUR and 100 USD to CHF.",
         lambda w, calls, reply: said(reply, 92.17, 88.5)),
    case("email-file", "sequential", "Email Priya the agenda from docs/agenda.md.",
         lambda w, calls, reply: any(mentions(email["body"], "Hiring plan", "Atlas v2")
                                     for email in emails_to(w, "priya.nair@example.com"))),
    case("event-contacts", "sequential", "Schedule a 45 minute 'Design review' with Alice and Bob tomorrow at 10:00.",
         lambda w, calls, reply: check_event(w, "Design review", "2026-09-12T10:00", 45, [ALICE, BOB])),
    case("invoice-total", "sequential", "What's the total amount across all the invoices in the invoices folder?",
         lambda w, calls, reply: said(reply, 4540.75)),
    case("config-search", "sequential", "Which config file sets the server port to 8080?",
         lambda w, calls, reply: mentions(reply, "dev.yml")),
    case("collect-todos", "sequential", "Collect the TODO comments from src/app.py into a new file notes/todo.md.",
         lambda w, calls, reply: mentions(w.state["files"].get("notes/todo.md"), "validate the upload size",
                                          "retry failed webhooks")),
    case("log-errors", "sequential", "Put every ERROR line from logs/app.log into logs/errors.txt, nothing else.",
         lambda w, calls, reply: all(line.split(" ERROR ")[1] in w.state["files"].get("logs/errors.txt", "")
                                     for line in ERROR_LINES)
         and " INFO " not in w.state["files"].get("logs/errors.txt", " INFO ")),
    case("undelivered-report", "sequential", "Email me a list of my orders that haven't been delivered yet.",
         lambda w, calls, reply: any(mentions(email["body"], "A1003", "A1005")
                                     and not re.search(r"A100[124]", email["body"])
                                     for email in emails_to(w, DANA))),
    case("currency-sum", "sequential", "I have 1200 JPY and 30 GBP. How many euros is that altogether?",
         lambda w, calls, reply: said(reply, 42.64)),
    case("invoice-in-euros", "sequential", "What is invoice INV-003's amount in euros?",
         lambda w, calls, reply: said(reply, 2129.26)),
    case("wrong-extension", "recovery", "What database host is configured in config/settings.yaml?",
         lambda w, calls, reply: mentions(reply, "db-prod-3.internal") and not mutated(calls)),
    case("flaky-service", "recovery", "Is it raining in London right now?",
         lambda w, calls, reply: succeeded(calls, "get_weather") and mentions(reply, "rain"),
         world={"failures": {"get_weather": 1}}),
    case("ambiguous-contact", "clarify", "Send Sam the link to the Q3 agenda: https://intra.example.com/q3",
         lambda w, calls, reply: not w.state["emails"] and asked(reply)),
    case("missing-time", "clarify", "Book a meeting with Alice.",
         lambda w, calls, reply: not new_events(w) and asked(reply)),
    case("missing-content", "clarify", "Send an email to Bob.",
         lambda w, calls, reply: not w.state["emails"] and asked(reply)),
    case("no-tool-stock", "irrelevant", "What's NVIDIA's stock price right now?",
         lambda w, calls, reply: not calls),
    case("no-tool-haiku", "irrelevant", "Write a haiku about autumn.", lambda w, calls, reply: not calls),
    case("no-tool-arithmetic", "irrelevant", "What is 17 multiplied by 23?",
         lambda w, calls, reply: not calls and said(reply, 391)),
    case("no-tool-thanks", "irrelevant", "Thanks, that's all for now!", lambda w, calls, reply: not calls),
    case("reminder-not-event", "distractor", "Remind me at 17:00 today to call my mom.",
         lambda w, calls, reply: not new_events(w) and any(
             reminder["at"] == "2026-09-11T17:00" and mentions(reminder["text"], "mom")
             for reminder in w.state["reminders"])),
    case("event-not-reminder", "distractor", "Add a one hour 'Dentist' appointment to my calendar on 2026-09-15 at 08:00.",
         lambda w, calls, reply: check_event(w, "Dentist", "2026-09-15T08:00", 60) and not w.state["reminders"]),
    case("refund-eligible", "policy", "Please refund order A1001, the grinder arrived broken.",
         lambda w, calls, reply: refunded(w) == {"A1001"}),
    case("refund-too-old", "policy", "I want a refund for order A1002.",
         lambda w, calls, reply: not refunded(w) and calls_to(calls, "get_order", order_id="A1002")),
    case("refund-not-delivered", "policy", "Refund order A1003 please.",
         lambda w, calls, reply: not refunded(w) and calls_to(calls, "get_order", order_id="A1003")),
    case("refund-all-eligible", "policy", "Refund every order of mine that is still eligible for a refund.",
         lambda w, calls, reply: refunded(w) == {"A1001", "A1004"}),
    case("cancel-one", "sequential", "Cancel my standup on Monday.",
         lambda w, calls, reply: event_ids(w) == {"E2", "E3", "E4", "E5"}),
    case("clear-except", "sequential", "Clear my calendar on 2026-09-16 except the meeting with Bob.",
         lambda w, calls, reply: event_ids(w) == {"E1", "E2", "E3"}),
    case("conditional-rain", "conditional", "Check the weather in London. If it's raining, email Alice that the picnic is off.",
         lambda w, calls, reply: any(mentions(email["body"] + email["subject"], "picnic") for email in emails_to(w, ALICE))),
    case("conditional-sunny", "conditional", "Check the weather in Madrid. If it's raining, email Alice that the picnic is off.",
         lambda w, calls, reply: calls_to(calls, "get_weather") and not w.state["emails"]),
    case("follow-up-unit", "multi-turn", "And in Stockholm?",
         lambda w, calls, reply: said(reply, 11) and not said(reply, 52), history=OSLO_HISTORY),
    case("follow-up-invite", "multi-turn", "Actually, invite Bob to it too.",
         lambda w, calls, reply: len(w.state["events"]) == 6
         and set(next(event for event in w.state["events"] if event["id"] == "E6")["attendees"]) == {ALICE, BOB},
         history=BUDGET_HISTORY, world=BUDGET_WORLD),
    case("follow-up-answer", "multi-turn", "Ortiz. The link is https://intra.example.com/q3",
         lambda w, calls, reply: [email["to"] for email in w.state["emails"]] == [["sam.ortiz@example.com"]]
         and mentions(w.state["emails"][0]["body"], "https://intra.example.com/q3"), history=SAM_HISTORY),
]
BY_ID = {entry["id"]: entry for entry in CASES}


def messages_for(entry):
    return [{"role": "system", "content": SYSTEM}, *entry["history"], {"role": "user", "content": entry["prompt"]}]


def grade(case_id, world, calls, reply):
    return bool(BY_ID[case_id]["check"](world, calls, reply or ""))
