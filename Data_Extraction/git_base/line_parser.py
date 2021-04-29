import re
import sys

FILE_DIFF_HEADER = re.compile(r"^diff --git a/(?P<from_file>.*?)\s* b/(?P<to_file>.*?)\s*$")
OLD_MODE_HEADER = re.compile(r"^old mode (?P<mode>\d+)$")
NEW_MODE_HEADER = re.compile(r"^new mode (?P<mode>\d+)$")
NEW_FILE_MODE_HEADER = re.compile(r"^new file mode (?P<mode>\d+)$")
DELETED_FILE_MODE_HEADER = re.compile(r"^deleted file mode (?P<mode>\d+)$")
INDEX_DIFF_HEADER = re.compile(r"^index (?P<from_blob>.*?)\.\.(?P<to_blob>.*?)(?: (?P<mode>\d+))?$")
BINARY_DIFF = re.compile(r"Binary files (?P<from_file>.*) and (?P<to_file>.*) differ$")
A_FILE_CHANGE_HEADER = re.compile(r"^--- (?:/dev/null|a/(?P<file>.*?)\s*)$")
B_FILE_CHANGE_HEADER = re.compile(r"^\+\+\+ (?:/dev/null|b/(?P<file>.*?)\s*)$")
CHUNK_HEADER = re.compile(r"^@@ -(?P<from_line_start>\d+)(?:,(?P<from_line_count>\d+))? \+(?P<to_line_start>\d+)(?:,(?P<to_line_count>\d+))? @@(?P<line>.*)$")
LINE_DIFF = re.compile(r"^(?P<action>[-+ ])(?P<line>.*)$")
NO_NEWLINE = re.compile(r"^\\ No newline at end of file$")
RENAME_HEADER = re.compile(r"^similarity index (?P<rate>\d*)")
RENAME_A_FILE = re.compile(r"^rename from (?P<from_file>.*?)")
RENAME_B_FILE = re.compile(r"^rename to (?P<to_file>.*?)")


class ParseError(Exception):
    pass

class LineParseError(Exception):
    def __init__(self, message, line):
        self.line = line
        super(LineParseError, self).__init__(message)

    def __str__(self):
        return "Line {}: {}".format(self.line, super(LineParseError, self).__str__())

def parse_lines(line_iterable):
    state = "start_of_file"
    for line_index, line in enumerate(line_iterable):
        prev_state = state
        try:
            state, parsed = parse_line(line, prev_state)
        except ParseError as parse_exc:
            raise LineParseError("{} ({!r})".format(parse_exc, line), line_index + 1)
        else:
            yield state, parsed, line

def parse_line(line, prev_state):
    # "diff --git a/{TO_FILE} b/{TO_FILE}""
    if prev_state in ("start_of_file", "new_mode_header", "line_diff", "no_newline", "index_diff_header", "binary_diff", "rename_b_file"):
        match = FILE_DIFF_HEADER.search(line)
        if match:
            return "file_diff_header", match.groupdict()
        elif prev_state == "start_of_file":
            raise ParseError("Expected file diff header")

    # "old mode {MODE}"
    if prev_state == "file_diff_header":
        match = OLD_MODE_HEADER.search(line)
        if match:
            return "old_mode_header", match.groupdict()

    # "new mode {MODE}"
    if prev_state == "old_mode_header":
        match = NEW_MODE_HEADER.search(line)
        if match:
            return "new_mode_header", match.groupdict()
        else:
            raise ParseError("Expected new_mode_header")

    # "new file mode {MODE}"
    if prev_state == "file_diff_header":
        match = NEW_FILE_MODE_HEADER.search(line)
        if match:
            return "new_file_mode_header", match.groupdict()

    # "deleted file mode {MODE}"
    if prev_state == "file_diff_header":
        match = DELETED_FILE_MODE_HEADER.search(line)
        if match:
            return "deleted_file_mode_header", match.groupdict()

    # "index {FROM_COMMIT} {TO_COMMIT} [{MODE}]"
    if prev_state in ("rename_b_file", "file_diff_header", "new_mode_header", "new_file_mode_header", "deleted_file_mode_header"):
        match = RENAME_HEADER.search(line)
        if match:
            return "rename_header", match.groupdict()
        match = INDEX_DIFF_HEADER.search(line)
        if match:
            return "index_diff_header", match.groupdict()
        else:
            raise ParseError("Expected index_diff_header")
    
    if prev_state == "rename_header":
        match = RENAME_A_FILE.search(line)
        if match:
            return "rename_a_file", match.groupdict()

    if prev_state == "rename_a_file":
        match = RENAME_B_FILE.search(line)
        if match:
            return "rename_b_file", match.groupdict()

    # "Binary files {FROM_FILE} and {TO_FILE} differ"
    if prev_state == "index_diff_header":
        match = BINARY_DIFF.search(line)
        if match:
            return "binary_diff", match.groupdict()

    # "--- {FILENAME}"
    if prev_state == "index_diff_header":
        match = A_FILE_CHANGE_HEADER.search(line)
        if match:
            return "a_file_change_header", match.groupdict()
        else:
            raise ParseError("Expected a_file_change_header")

    # "+++ {FILENAME}"
    if prev_state == "a_file_change_header":
        match = B_FILE_CHANGE_HEADER.search(line)
        if match:
            return "b_file_change_header", match.groupdict()
        else:
            raise ParseError("Expected b_file_change_header")

    # "@@ {?}[,{?}] {?}[,{?}] @@[{LINE}]"
    if prev_state in ("b_file_change_header", "line_diff", "no_newline"):
        match = CHUNK_HEADER.search(line)
        if match:
            parsed = match.groupdict()
            if parsed["from_line_count"] is None: parsed["from_line_count"] = 1
            if parsed["to_line_count"]   is None: parsed["to_line_count"]   = 1
            if parsed["to_line_start"]   is None: parsed["to_line_start"]   = parsed["from_line_start"]
            parsed["from_line_start"] = int(parsed["from_line_start"])
            parsed["from_line_count"] = int(parsed["from_line_count"])
            parsed["to_line_start"]   = int(parsed["to_line_start"])
            parsed["to_line_count"]   = int(parsed["to_line_count"])
            return "chunk_header", parsed
        elif prev_state == "b_file_change_header":
            raise ParseError("Expected chunk_header")

    # "-{LINE}"
    # "+{LINE}"
    # " {LINE}"
    if prev_state in ("chunk_header", "line_diff", "no_newline"):
        match = LINE_DIFF.search(line)
        if match:
            return "line_diff", match.groupdict()

    # "\ No newline at end of file"
    if prev_state in ("chunk_header", "line_diff"):
        match = NO_NEWLINE.search(line)
        if match:
            return "no_newline", match.groupdict()
        else:
            raise ParseError("Expected line_diff or no_newline")

    raise ParseError("Can't parse line with prev_state {!r}".format(prev_state))

if __name__ == "__main__":
    for state, parsed, line in parse_lines(sys.stdin):
        print(state, line.rstrip())
