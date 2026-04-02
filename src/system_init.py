from __future__ import annotations

from .commands import built_in_command_names, get_commands
from .context import build_port_context, render_context
from .setup import run_setup
from .tools import get_tools


def build_system_init_message(trusted: bool = True) -> str:
    setup = run_setup(trusted=trusted)
    commands = get_commands()
    tools = get_tools()
    context = build_port_context(include_git=True, include_tree=True)

    lines = [
        '# System Init',
        '',
        f'Trusted: {setup.trusted}',
        f'Built-in command names: {len(built_in_command_names())}',
        f'Loaded command entries: {len(commands)}',
        f'Loaded tool entries: {len(tools)}',
        '',
        '## Workspace Context',
        render_context(context, include_tree=False),
        '',
        'Startup steps:',
        *(f'- {step}' for step in setup.setup.startup_steps()),
    ]
    return '\n'.join(lines)
