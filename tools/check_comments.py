from io import StringIO
from pathlib import Path
import sys
import tokenize


def is_code_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped
        and not stripped.startswith("#")
        and not stripped.startswith("'''")
        and not stripped.startswith('"""')
    )


def is_inline_comment(line: str) -> bool:
    """Detect if line contains a real inline comment (not part of a string)."""
    try:
        tokens = list(tokenize.generate_tokens(StringIO(line).readline))
        for tok_type, tok_str, *_ in tokens:
            if tok_type == tokenize.COMMENT:
                return True
    except tokenize.TokenError:
        pass
    return False


def check_file(path: Path) -> list[tuple[int, str]]:
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return errors

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for inline comment (ignore comments in strings)
        if is_inline_comment(line) and not stripped.startswith("#"):
            if "# noqa" not in line and "# noqa" not in line:
                errors.append(
                    (i + 1, "Inline comment: move to its own line above.")
                )

        # Check for floating/unattached comment
        if stripped.startswith("#"):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line == "" and line != "# fmt: on":
                    errors.append(
                        (
                            i + 1,
                            "Floating comment: no blank lines between comment and code.",
                        )
                    )
                elif (
                    not next_line.startswith("#")
                    and not is_code_line(next_line)
                    and line != "# fmt: on"
                ):
                    errors.append(
                        (
                            i + 1,
                            "Unattached comment: must be followed by code.",
                        )
                    )
    return errors


def main() -> None:
    has_error = False
    for filename in sys.argv[1:]:
        path = Path(filename)
        if not path.suffix == ".py":
            continue
        errors = check_file(path)
        for line_num, msg in errors:
            print(f"{path}:{line_num}: {msg}")
            has_error = True
    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()

# Example usage:
# python precommit/check_comments.py path/to/file1.py path/to/file2.py
