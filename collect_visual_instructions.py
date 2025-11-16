from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Union

ROOT_DIR = Path(__file__).resolve().parent
TARGET_DIRS = [
    ROOT_DIR / "data" / "train_valid" / "train" / "press_json",
    ROOT_DIR / "data" / "train_valid" / "train" / "report_json",
]
OUTPUT_PATH = ROOT_DIR / "queries.txt"


def extract_instructions(node: Union[dict, list, str, int, float, None]) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "visual_instruction" and isinstance(value, str):
                yield value.strip()
            else:
                yield from extract_instructions(value)
    elif isinstance(node, list):
        for item in node:
            yield from extract_instructions(item)


def load_json(path: Path) -> Union[dict, list]:
    with path.open("r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        # Some dataset entries are intentionally blank – treat them as missing and skip.
        raise ValueError("empty file")

    return json.loads(content)


def gather_instructions(paths: Iterable[Path]) -> list[str]:
    collected: list[str] = []
    for directory in paths:
        if not directory.exists():
            print(f"[warn] directory not found: {directory}")
            continue

        for json_path in sorted(directory.rglob("*.json")):
            try:
                data = load_json(json_path)
            except ValueError as exc:
                print(f"[skip] {json_path}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - best effort extraction
                print(f"[warn] failed to parse {json_path}: {exc}")
                continue

            for instruction in extract_instructions(data):
                if instruction:
                    collected.append(instruction)
    return collected


def main() -> None:
    instructions = gather_instructions(TARGET_DIRS)
    OUTPUT_PATH.write_text("\n".join(instructions), encoding="utf-8")
    print(
        f"wrote {len(instructions)} visual instructions "
        f"from {len(TARGET_DIRS)} directories to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
