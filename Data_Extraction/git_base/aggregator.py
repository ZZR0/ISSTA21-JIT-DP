import line_parser

def aggregator(parsed_lines_iterable):
    def set_once(root, path, value):
        original_path = list(path)

        path = list(path)
        last_key = path.pop()

        for key in path:
            root = root[key]

        if last_key in root:
            raise KeyError("{!r} is already set".format(original_path))

        root[last_key] = value

    file_diff = None
    file_meta = None
    for state, parsed, _ in parsed_lines_iterable:
        if state == "file_diff_header":
            if file_diff is not None:
                yield file_diff
                file_diff = None
                file_meta = None

            file_diff = {}
            file_meta = {"no_newline_count": 0}
            set_once(file_diff, ("from",), {})
            set_once(file_diff, ("from", "file",), parsed["from_file"])
            set_once(file_diff, ("from", "end_newline",), True)
            set_once(file_diff, ("to",), {})
            set_once(file_diff, ("to", "file",), parsed["to_file"])
            set_once(file_diff, ("to", "end_newline",), True)
            set_once(file_diff, ("is_binary",), False)
            set_once(file_diff, ("chunks",),    [])

            set_once(file_diff, ("rename",), False)

            set_once(file_diff, ("meta_a",), {})
            set_once(file_diff, ("meta_b",), {})

            set_once(file_diff, ("meta_a", "name"), file_diff["from"]["file"])
            set_once(file_diff, ("meta_a", "lines"), 0)

            set_once(file_diff, ("meta_b", "name"), file_diff["to"]["file"])
            set_once(file_diff, ("meta_b", "lines"), 0)

            file_diff["content"] = []
            continue

        if state == "new_file_mode_header":
            set_once(file_diff, ("from", "mode",), "0000000")
            set_once(file_diff, ("to",   "mode",), parsed["mode"])
            continue

        if state == "old_mode_header":
            set_once(file_diff, ("from", "mode",), parsed["mode"])
            continue

        if state == "new_mode_header":
            set_once(file_diff, ("to", "mode",), parsed["mode"])
            continue

        if state == "deleted_file_mode_header":
            set_once(file_diff, ("from", "mode",), parsed["mode"])
            set_once(file_diff, ("to",   "mode",), "0000000")
            continue

        if state in ("a_file_change_header", "b_file_change_header"):
            key = {"a_file_change_header": "from", "b_file_change_header": "to"}[state]
            if file_diff[key]["file"] != parsed["file"] and parsed["file"] is not None:
                print(file_diff, parsed)
                raise Exception("TODO: Exception text")
            continue

        if state == "binary_diff":
            file_diff["is_binary"] = True
            continue

        if state == "rename_header":
            if "100" in parsed["rate"]:
                file_diff["rename"] = True
            continue

        if state == "rename_a_file":
            continue

        if state == "rename_b_file":
            continue

        if state == "index_diff_header":
            set_once(file_diff, ("from", "blob",), parsed["from_blob"])
            set_once(file_diff, ("to", "blob",), parsed["to_blob"])
            if parsed["mode"] is not None:
                set_once(file_diff, ("from", "mode",), parsed["mode"])
                set_once(file_diff, ("to", "mode"), parsed["mode"])
            continue

        if state == "chunk_header":
            file_diff["meta_a"]["lines"] += parsed["from_line_count"]
            file_diff["meta_b"]["lines"] += parsed["to_line_count"]
            continue

        if state == "line_diff":

            if parsed["action"] == " ":
                if file_diff["content"] and "ab" in file_diff["content"][-1]:
                    file_diff["content"][-1]["ab"].append(parsed["line"])
                else:
                    file_diff["content"].append({"ab": [parsed["line"]]})

            if parsed["action"] == "+":
                if file_diff["content"] and "ab" not in file_diff["content"][-1]:
                    if "b" in file_diff["content"][-1]:
                        file_diff["content"][-1]["b"].append(parsed["line"])
                    else:
                        file_diff["content"][-1]["b"] = [parsed["line"]]
                else:
                    file_diff["content"].append({"b": [parsed["line"]]})

            if parsed["action"] == "-":
                if file_diff["content"] and "ab" not in file_diff["content"][-1]:
                    if "a" in file_diff["content"][-1]:
                        file_diff["content"][-1]["a"].append(parsed["line"])
                    else:
                        file_diff["content"][-1]["a"] = [parsed["line"]]
                else:
                    file_diff["content"].append({"a": [parsed["line"]]})

            if file_meta["no_newline_count"] > 0:
                file_diff["to"]["end_newline"] = True
                file_diff["from"]["end_newline"] = False

            continue

        if state == "no_newline":
            file_meta["no_newline_count"] += 1
            if file_meta["no_newline_count"] > 2:
                raise Exception("TODO: Exception text")
            file_diff["to"]["end_newline"] = False
            continue

        raise Exception("Unexpected {!r} line".format(state))

    if file_diff is not None:
        yield file_diff
