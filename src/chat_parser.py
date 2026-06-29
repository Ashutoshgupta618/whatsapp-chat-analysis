import re
from pathlib import Path


def date_time(s: str) -> bool:
    pattern = r'^([0-9]+)(/)([0-9]+)(/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
    return bool(re.match(pattern, s))


def find_author(s: str) -> bool:
    parts = s.split(":")
    return len(parts) == 2


def get_datapoint(line: str):
    splitline = line.split(' - ')
    date_time_value = splitline[0]
    date, time = date_time_value.split(", ")
    message = " ".join(splitline[1:])

    if find_author(message):
        splitmessage = message.split(": ", 1)
        author = splitmessage[0]
        message = splitmessage[1]
    else:
        author = None

    return date, time, author, message


def parse_whatsapp_chat(file_path: str | Path):
    data = []
    file_path = Path(file_path)

    with file_path.open(encoding="utf-8") as fp:
        fp.readline()
        message_buffer = []
        date = time = author = None

        while True:
            line = fp.readline()
            if not line:
                break

            line = line.strip()
            if date_time(line):
                if len(message_buffer) > 0:
                    data.append([date, time, author, ' '.join(message_buffer)])
                message_buffer.clear()
                date, time, author, message = get_datapoint(line)
                message_buffer.append(message)
            else:
                message_buffer.append(line)

    return data
