SYSTEM = "You are a helpful assistant. Today is 2026-09-11. Use the tools when they are needed."


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    tool("get_weather", "Current weather for a city.", {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    }, ["city"]),
    tool("search_files", "Search file contents in a directory with a regex pattern.", {
        "pattern": {"type": "string"},
        "path": {"type": "string"},
    }, ["pattern", "path"]),
    tool("run_shell", "Run a shell command and return its output.", {
        "command": {"type": "string"},
        "timeout_s": {"type": "integer"},
    }, ["command"]),
    tool("create_event", "Create a calendar event.", {
        "title": {"type": "string"},
        "start": {"type": "string", "description": "ISO-8601 local date-time, e.g. 2026-09-12T10:00"},
        "duration_minutes": {"type": "integer"},
        "attendees": {"type": "array", "items": {"type": "string"}},
    }, ["title", "start", "duration_minutes"]),
    tool("convert_currency", "Convert an amount between ISO-4217 currencies.", {
        "amount": {"type": "number"},
        "from_currency": {"type": "string"},
        "to_currency": {"type": "string"},
    }, ["amount", "from_currency", "to_currency"]),
    tool("git_commit", "Stage the given files and commit them.", {
        "message": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"}},
    }, ["message", "files"]),
    tool("read_file", "Read a line range from a file.", {
        "path": {"type": "string"},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
    }, ["path"]),
]

CASES = [
    ("What's the weather in Helsinki right now? Use celsius.", [("get_weather", {"city": "Helsinki", "unit": "celsius"})]),
    ("Is it hot in Phoenix today? I think in fahrenheit.", [("get_weather", {"city": "Phoenix", "unit": "fahrenheit"})]),
    ("Find every TODO comment under the src directory.", [("search_files", {"pattern": "TODO", "path": "src"})]),
    ("Look for the string 'deprecated_api' in ./lib.", [("search_files", {"pattern": "deprecated_api", "path": "./lib"})]),
    ("Run `pytest -q` and give it at most 300 seconds.", [("run_shell", {"command": "pytest -q", "timeout_s": 300})]),
    ("Show me the current disk usage with df -h.", [("run_shell", {"command": "df -h"})]),
    (
        "Book a 45 minute 'Design review' on 2026-09-12T10:00 with alice@example.com and bob@example.com.",
        [("create_event", {"title": "Design review", "start": "2026-09-12T10:00", "duration_minutes": 45,
                           "attendees": ["alice@example.com", "bob@example.com"]})],
    ),
    ("Put a 30 minute 'Standup' on my calendar at 2026-09-14T09:30.",
     [("create_event", {"title": "Standup", "start": "2026-09-14T09:30", "duration_minutes": 30})]),
    ("How many US dollars is 250 euros?",
     [("convert_currency", {"amount": 250, "from_currency": "EUR", "to_currency": "USD"})]),
    ("Convert 1200 Japanese yen into British pounds.",
     [("convert_currency", {"amount": 1200, "from_currency": "JPY", "to_currency": "GBP"})]),
    ("Commit README.md and setup.py with the message 'Bump version to 1.4.0'.",
     [("git_commit", {"message": "Bump version to 1.4.0", "files": ["README.md", "setup.py"]})]),
    ("Show me lines 10 to 40 of app/main.py.", [("read_file", {"path": "app/main.py", "start_line": 10, "end_line": 40})]),
    ("Open config/settings.yaml.", [("read_file", {"path": "config/settings.yaml"})]),
    ("What's the weather in Tokyo and in Paris? Celsius for both.",
     [("get_weather", {"city": "Tokyo", "unit": "celsius"}), ("get_weather", {"city": "Paris", "unit": "celsius"})]),
    ("Convert 100 USD to EUR and 100 USD to CHF.",
     [("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}),
      ("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "CHF"})]),
    ("What is 17 multiplied by 23?", []),
    ("Explain in one sentence what a mutex is.", []),
    ("Write a haiku about autumn.", []),
    ("Thanks, that's all for now!", []),
    ("Which tool would you use to check tomorrow's forecast? Just tell me, don't call anything.", []),
]
