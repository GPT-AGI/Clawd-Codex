from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..protocol import ToolResult
from ..registry import ToolSpec


class AskUserQuestionTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="AskUserQuestion",
            description="Ask the user one or more multiple-choice questions.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},
                                "header": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "label": {"type": "string"},
                                            "description": {"type": "string"},
                                            "preview": {"type": "string"},
                                        },
                                        "required": ["label", "description"],
                                    },
                                },
                                "multiSelect": {"type": "boolean"},
                            },
                            "required": ["question", "header", "options"],
                        },
                    },
                    "answers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "annotations": {"type": "object"},
                    "metadata": {"type": "object"},
                },
                "required": ["questions"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        questions = tool_input.get("questions")
        if not isinstance(questions, list) or not (1 <= len(questions) <= 4):
            raise ToolInputError("questions must be an array of 1-4 items")
        self._validate_uniqueness(questions)

        answers = tool_input.get("answers")
        if answers is not None:
            if not isinstance(answers, dict) or not all(isinstance(v, str) for v in answers.values()):
                raise ToolInputError("answers must be an object mapping strings to strings when provided")
            out_answers = dict(answers)
        else:
            if context.ask_user is None:
                raise ToolPermissionError("AskUserQuestion requires user interaction but no ask_user handler is configured")
            out_answers = context.ask_user(questions)
            if not isinstance(out_answers, dict) or not all(isinstance(v, str) for v in out_answers.values()):
                raise ToolInputError("ask_user handler must return an object mapping question text to answer string")

        output: dict[str, Any] = {"questions": questions, "answers": out_answers}
        annotations = tool_input.get("annotations")
        if annotations is not None:
            output["annotations"] = annotations
        return ToolResult(name="AskUserQuestion", output=output)

    def _validate_uniqueness(self, questions: list[dict[str, Any]]) -> None:
        seen_questions: set[str] = set()
        for q in questions:
            if not isinstance(q, dict):
                raise ToolInputError("each question must be an object")
            qt = q.get("question")
            if not isinstance(qt, str) or not qt:
                raise ToolInputError("question.question must be a non-empty string")
            if qt in seen_questions:
                raise ToolInputError("question texts must be unique")
            seen_questions.add(qt)
            opts = q.get("options")
            if not isinstance(opts, list):
                raise ToolInputError("question.options must be an array")
            labels: set[str] = set()
            for opt in opts:
                if not isinstance(opt, dict):
                    raise ToolInputError("question.options[] must be objects")
                label = opt.get("label")
                if not isinstance(label, str) or not label:
                    raise ToolInputError("option.label must be a non-empty string")
                if label in labels:
                    raise ToolInputError("option labels must be unique within each question")
                labels.add(label)

